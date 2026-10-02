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
import hashlib
import sys

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import keras

import importlib
import dr_core as core
core = importlib.reload(core)        # always use the latest dr_core.py after a redeploy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="DR Stage Screening", layout="wide")


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


# ------------------------------- AI assistant (Gemini) ----------------------
# The assistant only receives the NUMBERS of the current result (never the image),
# and strict instructions so it explains the result instead of diagnosing.
CHAT_MODEL_DEFAULT = "gemini-2.5-flash"
SYSTEM_PROMPT = """You are a friendly assistant inside a university research prototype for
diabetic retinopathy (DR) screening. Rules:
1. Explain the screening result below in simple, clear language for a patient or a health worker.
   Keep answers short (about 150 words) unless the user asks for more detail.
2. Use only the result data below and well-established general knowledge about DR and diabetes.
   Never invent numbers, findings or test results.
3. You are not a doctor. Do not give a diagnosis, do not change the predicted stage or the
   suggested action, and do not recommend specific medicines, doses or treatments.
   Encourage confirmation by a qualified eye-care professional when it is relevant.
4. If the user mentions sudden vision loss, eye pain, flashes of light, many new floaters or a
   curtain over their vision, tell them to seek urgent medical care immediately.
5. Explain technical terms (stage names, probability, uncertainty, Grad-CAM, triage) simply.
6. If a question is not about eye health, DR, diabetes or this result, politely say that you can
   only help with those topics.
7. Mention that this is a research prototype and not a medical device when it is relevant."""

STAGE_INFO = ("Stages: 0 No DR (no visible damage); 1 Mild NPDR (microaneurysms only); "
              "2 Moderate NPDR (haemorrhages, hard exudates); 3 Severe NPDR (many haemorrhages, "
              "venous beading); 4 Proliferative DR (new abnormal vessels, highest risk). "
              "Stage 2 or higher is called referable DR.")


def get_secret(name, default=None):
    """Read a setting from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name, default)


@st.cache_resource
def get_chat_client(api_key, vertex=False):
    """Gemini client. Google AI Studio keys ('AQ.' authorization keys or older 'AIza' keys) use the
    Gemini API; vertex=True is only a fallback for Vertex AI express-mode keys."""
    from google import genai
    if vertex:
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)


def result_context(r):
    """Plain-text summary of the current result that is given to the assistant."""
    probs = ", ".join(f"{n} {p:.0%}" for n, p in zip(CLASS_NAMES, r["probs"]))
    return "\n".join([
        "CURRENT SCREENING RESULT (from the CNN model):",
        f"- Predicted stage: {CLASS_NAMES[r['stage']]} ({r['probs'][r['stage']]:.0%})",
        f"- Stage probabilities: {probs}",
        f"- Probability that any DR is present: {r['dr_prob']:.0%}",
        f"- Probability of referable DR (stage 2 or higher): {float(r['probs'][2:].sum()):.0%}",
        f"- Uncertainty: {r['entropy']:.2f} on a 0-1 scale (0 = very sure, 1 = completely unsure); "
        f"cases above {CFG['entropy_threshold']:.2f} are sent for human review",
        f"- Suggested action: {r['decision']}",
        "- Reasons: " + " ".join(r["reasons"]),
        "- Image quality warnings: " + (", ".join(r["warnings"]) if r["warnings"] else "none"),
        f"- Model: {CFG['backbone']} transfer-learning CNN; test accuracy {tm.get('acc', 0):.2f}, "
        f"QWK {tm.get('qwk', 0):.2f}. Grad-CAM shows the retina regions that influenced the prediction.",
        STAGE_INFO,
    ])


# models are tried in this order until one works; the "latest" aliases always point to
# Google's current Flash models, so the app keeps working when old model names are retired
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


def friendly_error(err):
    """Turn a Gemini error into a short message the user can act on."""
    e = str(err)
    if "API_KEY_INVALID" in e or "API key not valid" in e:
        return "The API key is not valid. Please check GEMINI_API_KEY in the app secrets."
    if "429" in e or "RESOURCE_EXHAUSTED" in e:
        return "The free usage limit was reached. Please wait a minute and try again."
    if "PERMISSION_DENIED" in e or "403" in e:
        return "The API key has no permission for the Gemini API (check the key's project in Google AI Studio)."
    if "location is not supported" in e.lower() or "FAILED_PRECONDITION" in e:
        return "The Gemini API is not available for this key's region or project."
    return "The assistant could not be reached right now. Please try again in a moment."


# models tried in this order (Google retires model names over time, so there is a fallback)
PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest",
                    "gemini-flash-lite-latest", "gemini-2.0-flash"]


@st.cache_data(ttl=3600, show_spinner=False)
def discover_flash_models(api_key):
    """Ask Google which 'flash' text models this API key can use right now."""
    try:
        found = []
        for m in get_chat_client(api_key).models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if ("generateContent" in actions and "flash" in name
                    and not any(x in name for x in ("image", "tts", "audio", "live", "embedding"))):
                found.append(name)
        return found
    except Exception as e:
        print("Model discovery failed:", repr(e))
        return []


def explain_error(e):
    """Turn an API error into a short, understandable message (full error goes to the logs)."""
    text, code = str(e), getattr(e, "code", None)
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "The API key is not valid. Please check GEMINI_API_KEY in the app secrets."
    if code == 401 or "UNAUTHENTICATED" in text or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in text:
        return ("The API key was rejected (401). Make sure requirements.txt asks for a recent google-genai "
                "version and that the key was copied completely.")
    if code == 403 or "PERMISSION_DENIED" in text:
        return "This API key is not allowed to use Gemini (permission denied)."
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return "The assistant is busy right now (free usage limit reached). Please try again in a minute."
    return f"Sorry, the assistant is not available at the moment (error {code or 'unknown'})."


def ask_assistant(question, history, r):
    """Send the question, the recent conversation and the result context to Gemini."""
    from google.genai import types
    api_key = (get_secret("GEMINI_API_KEY") or "").strip()
    # AI Studio now issues "AQ." authorization keys for the normal Gemini API, so the Gemini API is
    # always tried first; Vertex AI express mode is only a fallback if the key is rejected
    modes = [False, True]
    if "client_mode" in st.session_state:
        modes = [st.session_state["client_mode"]]
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in history[-10:]]                 # last 5 question/answer pairs only
    contents.append({"role": "user", "parts": [{"text": question}]})
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + "\n\n" + result_context(r),
                                         temperature=0.3, max_output_tokens=1024)

    # the model that worked last time first, then the configured one, preferred ones and discovered ones
    candidates = [st.session_state.get("working_model"), get_secret("GEMINI_MODEL")] + PREFERRED_MODELS
    candidates = [m for i, m in enumerate(candidates) if m and m not in candidates[:i]]
    last_error = None
    base_candidates = candidates
    for vertex in modes:
        client = get_chat_client(api_key, vertex)
        candidates = base_candidates
        result = _try_models(client, api_key, vertex, candidates, contents, config)
        if result is not None and result[0] == "ok":
            st.session_state["client_mode"] = vertex
            return result[1]
        last_error = result[1] if result else last_error
        # only an authentication / key problem is a reason to try the other key type
        if not _is_key_error(last_error):
            break
    return explain_error(last_error) if last_error else "Sorry, no Gemini model is available for this key."


def _is_key_error(e):
    text = str(e)
    return e is not None and (getattr(e, "code", None) in (400, 401, 403)
                              and ("API_KEY" in text or "API key" in text or "UNAUTHENTICATED" in text
                                   or "PERMISSION_DENIED" in text or "CREDENTIALS" in text.upper()))


def _try_models(client, api_key, vertex, candidates, contents, config):
    """Try the candidate models with one client. Returns ("ok", answer) or ("error", exception)."""
    last_error = None
    for attempt in (1, 2):
        for model in candidates:
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                st.session_state["working_model"] = model
                answer = (resp.text or "").strip()
                return ("ok", answer or "Sorry, I could not create an answer. Please try asking in a different way.")
            except Exception as e:
                print(f"Gemini error with model {model}:", repr(e))      # visible in 'Manage app' -> logs
                last_error = e
                text = str(e)
                # a wrong / retired model name or a model without free quota -> try the next one
                if getattr(e, "code", None) == 404 or "NOT_FOUND" in text or "limit: 0" in text:
                    continue
                return ("error", e)
        if attempt == 1 and not vertex:                  # none worked: ask Google which models exist
            extra = [m for m in discover_flash_models(api_key) if m not in candidates]
            if not extra:
                break
            candidates = extra
    return ("error", last_error) if last_error else None


# ------------------------------- user interface ------------------------------
st.title("Diabetic Retinopathy Stage Screening")
tm = CFG.get("test_metrics", {})
st.markdown("Check a retinal (fundus) photo for signs of diabetic retinopathy in three simple steps: "
            "**1. choose an image**, **2. read the result**, **3. ask questions** about it.")
with st.expander("About the model"):
    st.markdown(f"Transfer-learning **{CFG['backbone']}** network with three outputs (stage, DR yes/no and "
                "stage order), trained on the APTOS 2019 dataset. Each image is checked 20 times "
                "(4 flips x Monte-Carlo dropout) to measure how sure the model is, and Grad-CAM shows where "
                f"it looked. Test results: accuracy **{tm.get('acc', float('nan')):.3f}**, "
                f"macro-F1 **{tm.get('macro_f1', float('nan')):.3f}**, QWK **{tm.get('qwk', float('nan')):.3f}**.")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Image-quality check\n2. Image enhancement (CLAHE, mask-aware Ben Graham)\n"
                "3. CNN prediction with uncertainty\n4. Suggested follow-up (triage)\n"
                "5. Grad-CAM explanation\n6. AI assistant explains the result")
    st.markdown(DISCLAIMER)

ex_dir = os.path.join(BASE_DIR, "examples")
examples = sorted(os.listdir(ex_dir)) if os.path.isdir(ex_dir) else []

col_in, col_out = st.columns([1, 1.3], gap="large")
with col_in:
    st.subheader("1. Choose an image")
    uploaded = st.file_uploader("Upload a colour fundus photo (PNG or JPG)", type=["png", "jpg", "jpeg"])
    example = st.selectbox("Or try an example image", ["(none)"] + examples)
    image = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif example != "(none)":
        image = core.load_rgb(os.path.join(ex_dir, example))
    if image is not None:
        st.image(image, caption="Your image", width="stretch")

with col_out:
    st.subheader("2. Result")
    if image is None:
        st.info("Upload an image or pick an example on the left to see the result here.")
    else:
        # run the model only when a NEW image is chosen; chat messages reuse the stored result
        img_key = hashlib.md5(image.tobytes()).hexdigest()
        if st.session_state.get("img_key") != img_key:
            with st.spinner("Analysing the image ..."):
                st.session_state["result"] = analyze(image)
            st.session_state["img_key"] = img_key
            st.session_state["chat"] = []               # new image -> new conversation
        r = st.session_state["result"]
        if r["rejected"]:
            q = r["quality"]
            st.error("This does not look like a retinal (fundus) photo, so it was not analysed. "
                     "Please upload a colour photo of the back of the eye.\n\n"
                     f"Details: retina area {q['coverage']:.0%}, red/blue ratio {q['red_ratio']:.2f}.")
        else:
            with st.container(border=True):
                st.markdown(f"#### Predicted stage: {CLASS_NAMES[r['stage']]}")
                st.caption(f"Model confidence for this stage: {r['probs'][r['stage']]:.0%}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Signs of DR", "Yes" if r["dr_prob"] >= 0.5 else "No",
                          help=f"Probability that any DR is present: {r['dr_prob']:.0%}")
                m2.metric("Uncertainty", f"{r['entropy']:.2f}",
                          help=f"0 = very sure, 1 = completely unsure. Above {CFG['entropy_threshold']:.2f} "
                               "the case is sent for human review.")
                m3.metric("Image quality", "Good" if not r["warnings"] else "Check",
                          help="; ".join(r["warnings"]) if r["warnings"] else "No quality problems found.")
            box = st.warning if "REVIEW" in r["decision"] or "urgent" in r["decision"] else st.success
            box(f"**Suggested action:** {r['decision'].replace(' + HUMAN GRADER REVIEW', '')}"
                + (" - please have this checked by a specialist" if "REVIEW" in r["decision"] else "")
                + "\n\n" + "\n".join(f"- {x}" for x in r["reasons"]))
            st.markdown("**Probability of each stage**")
            st.bar_chart(pd.DataFrame({"probability": r["probs"]},
                                      index=[f"{i} - {n}" for i, n in enumerate(CLASS_NAMES)]),
                         horizontal=True, height=220)
            st.download_button("Download screening report", make_report(r),
                               file_name=f"dr_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")

if image is not None and not r["rejected"]:
    with st.expander("See what the model looked at", expanded=True):
        c1, c2 = st.columns(2)
        c1.image(r["proc"], caption="Enhanced image used by the model", width="stretch")
        c2.image(r["overlay"], caption="Grad-CAM: red areas influenced the result most", width="stretch")

st.divider()
st.subheader("3. Ask about your result")
if image is None or r["rejected"]:
    st.info("Analyse a fundus image first, then you can ask questions about the result here.")
elif not get_secret("GEMINI_API_KEY"):
    st.info("The AI assistant is not configured (add GEMINI_API_KEY to the app secrets).")
else:
    st.caption("The assistant (Google Gemini) explains this result in simple words. It cannot diagnose and "
               "does not replace an eye doctor. Only the numbers shown above are shared with it - never your image.")
    chat = st.session_state.setdefault("chat", [])
    quick = ["What does my result mean?", "Why this suggested action?",
             "What does uncertainty mean?", "What should I do next?"]
    pending = None
    for col, q in zip(st.columns(len(quick)), quick):
        if col.button(q, width="stretch"):
            pending = q
    box = st.container(border=True)
    with box:
        if not chat and not pending:
            st.caption("Choose a question above or type your own below.")
        for m in chat:
            with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user"
                                 else ":material/support_agent:"):
                st.markdown(m["content"])
    typed = st.chat_input("Type your question about the result", max_chars=500)
    question = typed or pending
    if question:
        with box:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(question)
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                with st.spinner("Thinking ..."):
                    answer = ask_assistant(question, chat, r)
                st.markdown(answer)
        chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if chat and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()

st.divider()
st.caption(DISCLAIMER)
