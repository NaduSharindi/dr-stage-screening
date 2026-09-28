"""
dr_core.py
==========
Shared, reusable logic for the Diabetic Retinopathy (DR) Stage Detection project.

The SAME module is imported by:
  * the Google Colab training notebook, and
  * the cloud-deployed Gradio app (Hugging Face Spaces),
so the preprocessing applied at inference time is guaranteed to be identical to
the preprocessing used during training (prevents "training/serving skew").

Contents
--------
1. Fundus image preprocessing (crop, square pad, resize, denoise, CLAHE,
   Ben Graham local-contrast / edge enhancement, circular mask)
2. Image-quality metrics + data-driven quality gate          (innovation)
3. Multi-task CNN builder (stage softmax + DR binary + ordinal heads) (innovation)
4. Ordinal decoding and probability fusion                   (innovation)
5. Test-time augmentation + Monte-Carlo dropout uncertainty  (innovation)
6. Grad-CAM explainability                                   (innovation)
7. Referral / triage logic                                   (innovation)
"""

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
# ---- dr_core.py part 2 ----
# ============================================================================
# 1. PREPROCESSING
# ============================================================================
def load_rgb(path):
    """Read an image from disk as an RGB uint8 array (OpenCV reads BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_black_border(img, tol=7, min_frac=0.02):
    """Remove the uninformative black frame around the circular retina.
    A pixel is foreground if its grey value > tol; a row/column is kept if at
    least `min_frac` of its pixels are foreground. The morphological opening and
    the fraction rule make the crop robust to sensor noise and JPEG artefacts
    in the 'black' background."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    rows = np.where(mask.mean(axis=1) > min_frac)[0]
    cols = np.where(mask.mean(axis=0) > min_frac)[0]
    if len(rows) < 10 or len(cols) < 10:          # almost fully dark -> return as is
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    """Zero-pad to a square so that resizing does not distort the round retina."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def apply_clahe(img, clip_limit=2.0, grid=8):
    """Contrast Limited Adaptive Histogram Equalisation on the L (lightness)
    channel of LAB space -> boosts local contrast without shifting colours."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def retina_mask(img, tol=7, erode_px=2):
    """Binary mask of the real retina area (1 = retina). Unlike a fixed circle it
    also follows the flat top/bottom edges of photos where the retina is cut off."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:                                      # keep the largest region (the retina)
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    k = 2 * erode_px + 1
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def ben_graham(img, sigma=None, mask=None):
    """Ben Graham's method (Kaggle DR 2015 winner): subtract the local average
    colour (a large Gaussian blur). Acts as a high-pass / unsharp mask that
    normalises illumination and enhances edges of vessels, microaneurysms,
    haemorrhages and exudates.
    With `mask`, the local average is computed from retina pixels only
    (mask-normalised convolution: blur(img*mask) / blur(mask)). This prevents
    the bright halo that a plain blur creates wherever the retina meets the
    black background (circle edge and cut-off top/bottom edges)."""
    if sigma is None:
        sigma = img.shape[0] / 30.0
    if mask is None:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 4, blur, -4, 128)
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * m[:, :, None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[:, :, None]
    blur = num / np.maximum(den, 1e-3)
    out = 4.0 * img.astype(np.float32) - 4.0 * blur + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def basic_resize(img, size=IMG_SIZE):
    """Minimal pipeline (crop + square + resize). Used as the 'raw' baseline
    in the preprocessing ablation experiment."""
    img = pad_to_square(crop_black_border(img))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_fundus(img, size=IMG_SIZE, return_stages=False):
    """Full enhancement pipeline applied to every image (train, val, test and
    in the deployed app).

    Order: crop border -> pad square -> resize -> retina mask -> median denoise
           -> CLAHE -> mask-aware Ben Graham enhancement -> apply retina mask
    """
    stages = {"1. Original": img}
    img = crop_black_border(img);                    stages["2. Border cropped"] = img
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    stages["3. Square + resized"] = img
    mask = retina_mask(img)
    img = cv2.medianBlur(img, 3);                    stages["4. Median denoised"] = img
    img = apply_clahe(img);                          stages["5. CLAHE contrast"] = img
    img = ben_graham(img, mask=mask);                stages["6. Ben Graham (mask-aware)"] = img
    img = img * mask[:, :, None];                    stages["7. Retina mask applied"] = img
    return (img, stages) if return_stages else img


def prepare_input(img, size=IMG_SIZE, mode="enhanced"):
    """Model input for one RGB image: 'enhanced' = full pipeline, 'raw' = crop +
    resize only. The mode is chosen by the preprocessing ablation experiment."""
    return preprocess_fundus(img, size) if mode == "enhanced" else basic_resize(img, size)


def preprocess_path(path, size=IMG_SIZE):
    """Convenience wrapper returning (raw_resized, enhanced) for one file."""
    rgb = load_rgb(path)
    return basic_resize(rgb, size), preprocess_fundus(rgb, size)
# ---- dr_core.py part 3 ----
# ============================================================================
# 2. IMAGE QUALITY METRICS + QUALITY GATE
# ============================================================================
def image_quality_metrics(img):
    """Objective quality descriptors of an RGB image.
    * brightness : mean grey level of the retina area
    * contrast   : RMS contrast (std of grey levels)
    * sharpness  : variance of the Laplacian (edge energy / focus)
    * entropy    : Shannon entropy of the grey histogram (information content)
    * coverage   : fraction of the frame occupied by the retina
    * red_ratio  : mean(R) / mean(B) on the retina (fundus images are red/orange)
    """
    small = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    vals = gray[mask] if mask.sum() > 100 else gray.ravel()
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    r = small[..., 0][mask].mean() if mask.any() else 0.0
    b = small[..., 2][mask].mean() if mask.any() else 1.0
    return {
        "brightness": float(vals.mean()),
        "contrast": float(vals.std()),
        "sharpness": float(lap[mask].var() if mask.any() else lap.var()),
        "entropy": float(-(p * np.log2(p)).sum()),
        "coverage": float(mask.mean()),
        "red_ratio": float(r / (b + 1e-6)),
    }


def quality_gate(img, thresholds):
    """Data-driven quality gate. `thresholds` are percentiles computed on the
    TRAINING images in the notebook (so they reflect real acceptable images).
    Returns (is_fundus, warnings_list, metrics)."""
    m = image_quality_metrics(img)
    is_fundus = (m["coverage"] >= thresholds["coverage_min"]
                 and m["red_ratio"] >= thresholds["red_ratio_min"])
    warnings = []
    if m["brightness"] < thresholds["brightness_min"]:
        warnings.append("Image is too dark")
    if m["brightness"] > thresholds["brightness_max"]:
        warnings.append("Image is over-exposed")
    if m["sharpness"] < thresholds["sharpness_min"]:
        warnings.append("Image appears blurred / out of focus")
    if m["contrast"] < thresholds["contrast_min"]:
        warnings.append("Very low contrast")
    return is_fundus, warnings, m
# ---- dr_core.py part 4 ----
# ============================================================================
# 3. MULTI-TASK TRANSFER-LEARNING MODEL
# ============================================================================
def get_backbones():
    """Candidate ImageNet backbones and their matching input-normalisation."""
    from keras import applications as A
    return {
        "EfficientNetB0": (A.EfficientNetB0, A.efficientnet.preprocess_input),
        "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
        "DenseNet121": (A.DenseNet121, A.densenet.preprocess_input),
        "MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
    }


def get_preprocess_fn(backbone_name):
    return get_backbones()[backbone_name][1]


def build_model(backbone_name="EfficientNetB0", img_size=IMG_SIZE,
                dropout=0.4, dense_units=256, weights="imagenet"):
    """Three-headed network sharing one pretrained backbone:
      * stage   : 5-way softmax  (the required DR stage classifier)
      * dr      : 1 sigmoid      (DR present vs absent - binary screening)
      * ordinal : 4 sigmoids     (P(stage > k), k=0..3; encodes that stages
                                  are ORDERED, so 'Mild vs Severe' is a bigger
                                  error than 'Mild vs Moderate')
    The backbone is built with input_tensor so all its layers live in the main
    graph -> Grad-CAM can reach the last convolutional feature map directly.
    """
    import keras
    from keras import layers
    ctor, _ = get_backbones()[backbone_name]
    inp = keras.Input((img_size, img_size, 3), name="image")
    base = ctor(include_top=False, weights=weights, input_tensor=inp)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="drop_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="shared_dense")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)
    outputs = {
        "stage": layers.Dense(NUM_CLASSES, activation="softmax", name="stage")(x),
        "dr": layers.Dense(1, activation="sigmoid", name="dr")(x),
        "ordinal": layers.Dense(NUM_CLASSES - 1, activation="sigmoid", name="ordinal")(x),
    }
    model = keras.Model(inp, outputs, name=f"DR_multitask_{backbone_name}")
    return model, base


def last_conv_layer_name(model):
    """Name of the last layer that outputs a 4-D feature map (for Grad-CAM)."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No 4-D layer found")


def make_targets(y):
    """Integer stages -> dict of targets for the three heads."""
    y = np.asarray(y).astype(int)
    return {
        "stage": np.eye(NUM_CLASSES, dtype=np.float32)[y],
        "dr": (y > 0).astype(np.float32)[:, None],
        "ordinal": (y[:, None] > np.arange(NUM_CLASSES - 1)[None, :]).astype(np.float32),
    }
# ---- dr_core.py part 5 ----
# ============================================================================
# 4. ORDINAL DECODING + FUSION
# ============================================================================
def ordinal_to_probs(cum):
    """Convert cumulative probabilities P(y>k) into per-class probabilities.
    Monotonicity P(y>0) >= P(y>1) >= ... is enforced first."""
    cum = np.minimum.accumulate(np.asarray(cum, dtype=np.float64), axis=1)
    p = np.concatenate([1.0 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], axis=1)
    p = np.clip(p, 0.0, None)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def fuse_probs(stage_probs, ordinal_cum, w=0.5):
    """Weighted fusion of the softmax head and the ordinal head.
    w is tuned on the validation set to maximise Quadratic Weighted Kappa."""
    return w * np.asarray(stage_probs) + (1.0 - w) * ordinal_to_probs(ordinal_cum)
# ---- dr_core.py part 6 ----
# ============================================================================
# 5. TEST-TIME AUGMENTATION + MONTE-CARLO DROPOUT
# ============================================================================
def tta_views(x):
    """4 flip views of a batch (N,H,W,3). Fundus images are flip-invariant."""
    return [np.ascontiguousarray(v) for v in
            (x, x[:, :, ::-1], x[:, ::-1, :], x[:, ::-1, ::-1])]


def predict_tta(model, x, batch_size=64):
    """Deterministic prediction averaged over the 4 flip views."""
    outs = [model.predict(v, batch_size=batch_size, verbose=0) for v in tta_views(x)]
    return {k: np.mean([o[k] for o in outs], axis=0) for k in outs[0]}


def mc_dropout_predict(model, x, n_samples=10, fusion_w=0.5, use_tta=True, batch_size=32):
    """Monte-Carlo dropout (Gal & Ghahramani, 2016): run the network several
    times with dropout ACTIVE; the spread of predictions estimates model
    (epistemic) uncertainty. Frozen BatchNorm layers stay in inference mode.
    Returns mean fused probs (N,5), normalised predictive entropy (N,) in [0,1],
    and mean DR probability (N,)."""
    probs_all, dr_all = [], []
    for s in range(0, len(x), batch_size):
        xb = x[s:s + batch_size]
        views = tta_views(xb) if use_tta else [np.ascontiguousarray(xb)]
        fused, drp = [], []
        for _ in range(n_samples):
            for v in views:
                out = model(v, training=True)
                fused.append(fuse_probs(np.asarray(out["stage"]),
                                        np.asarray(out["ordinal"]), fusion_w))
                drp.append(np.asarray(out["dr"])[:, 0])
        probs_all.append(np.mean(fused, axis=0))
        dr_all.append(np.mean(drp, axis=0))
    p = np.concatenate(probs_all)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(NUM_CLASSES)
    return p, ent, np.concatenate(dr_all)
# ---- dr_core.py part 7 ----
# ============================================================================
# 6. GRAD-CAM
# ============================================================================
def grad_cam(model, x_single, conv_layer=None, class_idx=None):
    """Gradient-weighted Class Activation Map (Selvaraju et al., 2017) for the
    'stage' head. x_single: (1,H,W,3) already backbone-normalised.
    Returns a heat-map in [0,1] with the input's spatial size."""
    import tensorflow as tf
    import keras
    conv_layer = conv_layer or last_conv_layer_name(model)
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(conv_layer).output,
                              model.get_layer("stage").output])
    x_t = tf.convert_to_tensor(x_single, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_t, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                     # (1,C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-12)
    return cv2.resize(cam, (x_single.shape[2], x_single.shape[1]))


def overlay_cam(img_uint8, cam, alpha=0.4):
    """Colour the heat-map over the retina only (the black background stays black)."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img_uint8, 1 - alpha, heat, alpha, 0)
    background = img_uint8.max(axis=2) == 0
    out[background] = 0
    return out
# ---- dr_core.py part 8 ----
# ============================================================================
# 7. REFERRAL / TRIAGE LOGIC
# ============================================================================
# Illustrative follow-up pathway loosely based on common screening guidance.
# NOT clinical advice - local protocols must always be followed.
PATHWAY = {
    0: ("Routine", "No apparent DR. Routine re-screening in about 12 months."),
    1: ("Routine / monitor", "Mild NPDR. Re-screen in about 6-12 months; optimise glucose and blood pressure control."),
    2: ("Refer (non-urgent)", "Moderate NPDR. Ophthalmology review suggested within about 3-6 months."),
    3: ("Refer (urgent)", "Severe NPDR. Prompt ophthalmology referral suggested."),
    4: ("Refer (urgent)", "Proliferative DR. Urgent ophthalmology referral suggested."),
}


def triage(stage, entropy, dr_prob, entropy_threshold, quality_warnings=(),
           referable_prob=None, referable_threshold=None):
    """Combine prediction, uncertainty, head agreement, a referable-DR safety rule
    and image quality into one actionable decision. Returns (decision, reasons)."""
    level, text = PATHWAY[int(stage)]
    reasons = [text]
    manual = False
    if entropy > entropy_threshold:
        manual = True
        reasons.append(f"Model uncertainty is high ({entropy:.2f} > {entropy_threshold:.2f}).")
    if (dr_prob >= 0.5) != (int(stage) > 0):
        manual = True
        reasons.append("Binary DR head and stage head disagree.")
    if (referable_threshold is not None and referable_prob is not None
            and int(stage) < 2 and referable_prob >= referable_threshold):
        manual = True
        reasons.append(f"Safety rule: probability of referable DR (stage >= 2) is {referable_prob:.0%}, "
                       f"above the {referable_threshold:.0%} screening threshold.")
    if quality_warnings:
        manual = True
        reasons.append("Image quality issues: " + "; ".join(quality_warnings) + ".")
    decision = level + (" + HUMAN GRADER REVIEW" if manual else "")
    return decision, reasons
