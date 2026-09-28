"""
streamlit_app.py - Cloud-deployed Diabetic Retinopathy screening prototype
Hosted free on Streamlit Community Cloud (from a GitHub repository).
Research/education prototype only - NOT a medical device.

Pipeline for every uploaded image:
  quality gate -> same preprocessing as training (dr_core) -> multi-task CNN
  with TTA + Monte-Carlo dropout -> fused stage probabilities -> uncertainty
  -> Grad-CAM explanation -> triage decision -> downloadable report
"""
import os
import json
import datetime

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import keras

import importlib
import dr_core as core
core = importlib.reload(core)        # always use the latest dr_core.py after a redeploy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="DR Stage Screening", page_icon="👁️", layout="wide")


def _mtime(name):
    return os.path.getmtime(os.path.join(BASE_DIR, name))


@st.cache_resource(show_spinner="Loading the model ...")
def load_resources(config_mtime, model_mtime):
    """Load config and model once; Streamlit keeps them in memory between users.
    The file modification times are part of the cache key, so a redeployed
    model/config is loaded automatically instead of the old cached one."""
    with open(os.path.join(BASE_DIR, "config.json")) as f:
        cfg = json.load(f)
    model = keras.models.load_model(os.path.join(BASE_DIR, "dr_model.keras"), compile=False)
    return cfg, model


CFG, MODEL = load_resources(_mtime("config.json"), _mtime("dr_model.keras"))
PREPROCESS = core.get_preprocess_fn(CFG["backbone"])
CLASS_NAMES = CFG["class_names"]
DISCLAIMER = ("**Disclaimer:** research and education prototype built for a university coursework. "
              "It is NOT a medical device and must not be used for diagnosis. "
              "Always consult a qualified eye-care professional.")


def analyze(image):
    """Run the full screening pipeline on one RGB uint8 image. Returns a result dict."""
    is_fundus, warnings, qm = core.quality_gate(image, CFG["quality_thresholds"])
    if not is_fundus:
        return {"rejected": True, "quality": qm}

    proc = core.prepare_input(image, CFG["img_size"], CFG.get("input_mode", "enhanced"))
    x = PREPROCESS(proc.astype("float32")[None].copy())
    probs, ent, dr_prob = core.mc_dropout_predict(
        MODEL, x, n_samples=CFG["mc_samples"], fusion_w=CFG["fusion_w"])
    probs, ent, dr_prob = probs[0], float(ent[0]), float(dr_prob[0])
    stage = int(np.argmax(probs))
    cam = core.grad_cam(MODEL, x, CFG["conv_layer"], class_idx=stage)
    decision, reasons = core.triage(stage, ent, dr_prob, CFG["entropy_threshold"], warnings,
                                    referable_prob=float(probs[2:].sum()),
                                    referable_threshold=CFG.get("referable_threshold"))
    return {"rejected": False, "proc": proc, "overlay": core.overlay_cam(proc, cam),
            "probs": probs, "stage": stage, "entropy": ent, "dr_prob": dr_prob,
            "decision": decision, "reasons": reasons, "warnings": warnings, "quality": qm}


def make_report(r):
    lines = ["Diabetic Retinopathy Screening Report (prototype)",
             f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Model: {CFG['backbone']} multi-task transfer-learning CNN", ""]
    lines += [f"{n:<18}{p:6.1%}" for n, p in zip(CLASS_NAMES, r["probs"])]
    lines += ["", f"Predicted stage : {CLASS_NAMES[r['stage']]}",
              f"DR probability  : {r['dr_prob']:.3f}", f"Uncertainty     : {r['entropy']:.3f}",
              f"Decision        : {r['decision']}"]
    lines += [f"  - {x}" for x in r["reasons"]]
    lines += ["", "Not a medical device. For research/education only."]
    return "\n".join(lines)


# ------------------------------- user interface ------------------------------
st.title("👁️ Diabetic Retinopathy Stage Screening")
tm = CFG.get("test_metrics", {})
st.caption(f"Transfer-learning **{CFG['backbone']}** with multi-task heads (stage + binary + ordinal), "
           "test-time augmentation, Monte-Carlo-dropout uncertainty, Grad-CAM and automatic triage. "
           f"Test QWK **{tm.get('qwk', float('nan')):.3f}** · macro-F1 **{tm.get('macro_f1', float('nan')):.3f}** "
           f"· accuracy **{tm.get('acc', float('nan')):.3f}**")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Image-quality gate\n2. Crop + enhancement (CLAHE, mask-aware Ben Graham)\n"
                "3. CNN prediction (4 flips × Monte-Carlo dropout)\n4. Uncertainty + triage\n"
                "5. Grad-CAM explanation")
    st.markdown(DISCLAIMER)

ex_dir = os.path.join(BASE_DIR, "examples")
examples = sorted(os.listdir(ex_dir)) if os.path.isdir(ex_dir) else []

col_in, col_out = st.columns([1, 1.3])
with col_in:
    uploaded = st.file_uploader("Upload a colour fundus image", type=["png", "jpg", "jpeg"])
    example = st.selectbox("...or try an example image", ["(none)"] + examples)
    image = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif example != "(none)":
        image = core.load_rgb(os.path.join(ex_dir, example))
    if image is not None:
        st.image(image, caption="Input image", width="stretch")

with col_out:
    if image is None:
        st.info("Upload an image or pick an example to start.")
    else:
        with st.spinner("Analysing (quality check, prediction, uncertainty, Grad-CAM) ..."):
            r = analyze(image)
        if r["rejected"]:
            q = r["quality"]
            st.error(f"Image rejected: this does not look like a retinal fundus photograph "
                     f"(retina coverage {q['coverage']:.0%}, red/blue ratio {q['red_ratio']:.2f}).")
        else:
            st.subheader(f"Predicted stage: {CLASS_NAMES[r['stage']]} ({r['probs'][r['stage']]:.1%})")
            m1, m2, m3 = st.columns(3)
            m1.metric("DR present (binary head)", "Yes" if r["dr_prob"] >= 0.5 else "No", f"p = {r['dr_prob']:.2f}",
                      delta_color="off")
            m2.metric("Uncertainty", f"{r['entropy']:.2f}", f"threshold {CFG['entropy_threshold']:.2f}",
                      delta_color="off")
            m3.metric("Image quality", "OK" if not r["warnings"] else "Warning")
            st.bar_chart(pd.DataFrame({"probability": r["probs"]},
                                      index=[f"{i} - {n}" for i, n in enumerate(CLASS_NAMES)]),
                         horizontal=True)
            box = st.warning if "REVIEW" in r["decision"] or "urgent" in r["decision"] else st.success
            box(f"**Suggested action:** {r['decision']}\n\n" + "\n".join(f"- {x}" for x in r["reasons"]))
            st.download_button("Download screening report", make_report(r),
                               file_name=f"dr_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")

if image is not None and not r["rejected"]:
    c1, c2 = st.columns(2)
    c1.image(r["proc"], caption="Model input (after preprocessing)", width="stretch")
    c2.image(r["overlay"], caption="Grad-CAM: regions driving the decision", width="stretch")

st.markdown("---")
st.markdown(DISCLAIMER)
