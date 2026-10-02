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
import html
import time

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
# plus strict instructions so it explains the result instead of diagnosing.
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

# models tried in this order; Google retires old names, so the app also asks Google which exist
PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest",
                    "gemini-flash-lite-latest"]


def get_secret(name, default=None):
    """Read a setting from Streamlit secrets (cloud) or environment variables (local)."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name, default)


@st.cache_resource
def get_chat_client(api_key, vertex=False):
    """Gemini client. AI Studio keys ('AQ.' authorization keys or older 'AIza' keys) use the
    Gemini API; vertex=True is only a fallback for Vertex AI express-mode keys."""
    from google import genai
    if vertex:
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)


def result_context(r):
    """Plain-text summary of the current result that is given to the assistant."""
    tm = CFG.get("test_metrics", {})
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


def _code(e):
    return getattr(e, "code", None)


def _is_temporary(e):
    """Server overloaded (500/503) - usually disappears after a few seconds."""
    t = str(e)
    return _code(e) in (500, 502, 503, 504) or "UNAVAILABLE" in t or "overloaded" in t.lower()


def _is_model_problem(e):
    """Wrong/retired model name or no free quota for this model - try another model."""
    t = str(e)
    return _code(e) == 404 or "NOT_FOUND" in t or "limit: 0" in t


def _is_key_error(e):
    t = str(e)
    return e is not None and _code(e) in (400, 401, 403) and (
        "API_KEY" in t or "API key" in t or "UNAUTHENTICATED" in t or "PERMISSION_DENIED" in t)


def explain_error(e):
    """Turn an API error into a short, understandable message (full error goes to the logs)."""
    t, code = str(e), _code(e)
    if "API_KEY_INVALID" in t or "API key not valid" in t:
        return "The API key is not valid. Please check GEMINI_API_KEY in the app secrets."
    if code == 401 or "UNAUTHENTICATED" in t:
        return ("The API key was rejected (401). Make sure requirements.txt asks for google-genai>=2.27.0 "
                "and that the key was copied completely.")
    if code == 403 or "PERMISSION_DENIED" in t:
        return "This API key is not allowed to use Gemini (permission denied)."
    if _is_temporary(e):
        return "Google's AI service is very busy right now. Please wait a few seconds and ask again."
    if code == 429 or "RESOURCE_EXHAUSTED" in t:
        return "The free usage limit was reached. Please wait a minute and ask again."
    return f"Sorry, the assistant is not available at the moment (error {code or 'unknown'})."


def _try_models(client, api_key, vertex, candidates, contents, config):
    """Try the candidate models with one client. Returns ("ok", answer), ("error", e) or None."""
    last_error = None
    for attempt in (1, 2):
        for model in candidates:
            for retry in range(2):                       # every model: first try + one retry
                try:
                    resp = client.models.generate_content(model=model, contents=contents, config=config)
                    st.session_state["working_model"] = model
                    answer = (resp.text or "").strip()
                    return ("ok", answer or "Sorry, I could not create an answer. Please ask in a different way.")
                except Exception as e:
                    print(f"Gemini error with model {model}:", repr(e))   # visible in 'Manage app' -> logs
                    last_error = e
                    if _is_temporary(e) and retry == 0:
                        time.sleep(2)                    # busy server: wait a moment, retry once
                        continue
                    if _is_temporary(e) or _is_model_problem(e) or _code(e) == 429:
                        break                            # still busy / wrong model / no quota: next model
                    return ("error", e)                  # e.g. key problem: stop here
        if attempt == 1 and not vertex:                  # none worked: ask Google which models exist
            extra = [m for m in discover_flash_models(api_key) if m not in candidates]
            if not extra:
                break
            candidates = extra
    return ("error", last_error) if last_error else None


def ask_assistant(question, history, r):
    """Send the question, the recent conversation and the result context to Gemini."""
    from google.genai import types
    api_key = (get_secret("GEMINI_API_KEY") or "").strip()
    modes = [st.session_state["client_mode"]] if "client_mode" in st.session_state else [False, True]
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in history[-10:]]                 # last 5 question/answer pairs only
    contents.append({"role": "user", "parts": [{"text": question}]})
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + "\n\n" + result_context(r),
                                         temperature=0.3, max_output_tokens=1024)
    candidates = [st.session_state.get("working_model"), get_secret("GEMINI_MODEL")] + PREFERRED_MODELS
    candidates = [m for i, m in enumerate(candidates) if m and m not in candidates[:i]]
    last_error = None
    for vertex in modes:
        result = _try_models(get_chat_client(api_key, vertex), api_key, vertex, candidates, contents, config)
        if result and result[0] == "ok":
            st.session_state["client_mode"] = vertex
            return result[1]
        last_error = result[1] if result else last_error
        if not _is_key_error(last_error):                # only a key problem -> try the other key type
            break
    return explain_error(last_error) if last_error else "Sorry, no Gemini model is available for this key."


# ------------------------------- user interface ------------------------------
# colour of every stage: green (healthy) -> red (most severe)
STAGE_COLORS = ["#2e7d32", "#7cb342", "#f9a825", "#ef6c00", "#c62828"]
STAGE_TEXT = ["No signs of diabetic retinopathy were found.",
              "Very early changes (tiny bulges in small vessels) may be present.",
              "Clear signs of retinal damage are visible; an eye specialist should review it.",
              "Many signs of damage are visible; prompt specialist care is needed.",
              "The most advanced stage, with abnormal new vessels; urgent specialist care is needed."]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #e8f4fd 0%, #f3ecfd 100%);}
.hero {background: linear-gradient(120deg, #0f766e 0%, #2563eb 60%, #7c3aed 100%);
       color: #ffffff; padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
       box-shadow: 0 6px 18px rgba(37, 99, 235, 0.18);}
.hero h1 {color: #ffffff; margin: 0 0 6px 0; padding: 0; font-size: 2.1rem;}
.hero p {color: #e0f2fe; margin: 0; font-size: 1.05rem;}
.steps {display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap;}
.steps span {background: rgba(255,255,255,0.18); color: #ffffff; padding: 6px 14px;
             border-radius: 999px; font-size: 0.95rem;}
.sec {display: flex; align-items: center; gap: 10px; margin: 6px 0 12px 0;
      font-size: 1.3rem; font-weight: 700; color: #1e293b;}
.sec b {background: #2563eb; color: #ffffff; width: 32px; height: 32px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; font-size: 1rem;}
.card {border-radius: 16px; padding: 18px 20px; margin-bottom: 14px; color: #1e293b;
       background: #ffffff; box-shadow: 0 2px 10px rgba(15,23,42,0.08);}
.stage-name {font-size: 1.7rem; font-weight: 800; margin: 2px 0;}
.muted {color: #64748b; font-size: 0.9rem;}
.tiles {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0 2px 0;}
.tile {border-radius: 12px; padding: 10px 12px; background: #f8fafc; border: 1px solid #e2e8f0;}
.tile .t {color: #64748b; font-size: 0.82rem;}
.tile .v {font-size: 1.25rem; font-weight: 700;}
.bar-row {display: flex; align-items: center; gap: 10px; margin: 7px 0;}
.bar-label {width: 150px; font-size: 0.9rem; color: #334155;}
.bar-track {flex: 1; background: #eef2f7; border-radius: 999px; height: 14px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-pct {width: 46px; text-align: right; font-size: 0.9rem; font-weight: 600; color: #334155;}
.action {border-radius: 14px; padding: 14px 18px; margin: 4px 0 14px 0; border-left: 6px solid; color: #1e293b;}
.action h4 {margin: 0 0 6px 0; padding: 0;}
.action ul {margin: 6px 0 0 0; padding-left: 20px;}
.stButton > button, .stDownloadButton > button {border-radius: 999px; border: 1.5px solid #2563eb;
       color: #1d4ed8; background: #eff6ff; font-weight: 600;}
.stButton > button:hover, .stDownloadButton > button:hover {background: #2563eb; color: #ffffff;}
[data-testid="stFileUploader"] section {background: #f0f9ff; border: 2px dashed #7dd3fc; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

tm = CFG.get("test_metrics", {})
st.markdown("""
<div class="hero">
  <h1>Diabetic Retinopathy Screening</h1>
  <p>Check a retinal (fundus) photo for signs of diabetic eye disease and get an easy-to-understand result.</p>
  <div class="steps"><span>1&nbsp; Choose an image</span><span>2&nbsp; Read the result</span>
  <span>3&nbsp; Ask the assistant</span></div>
</div>""", unsafe_allow_html=True)


def section(num, title):
    st.markdown(f'<div class="sec"><b>{num}</b>{title}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("How it works")
    st.markdown("1. Image-quality check\n2. Image enhancement (CLAHE, mask-aware Ben Graham)\n"
                "3. CNN prediction with uncertainty\n4. Suggested follow-up (triage)\n"
                "5. Grad-CAM explanation\n6. AI assistant explains the result")
    with st.expander("About the model"):
        st.markdown(f"Transfer-learning **{CFG['backbone']}** with three outputs (stage, DR yes/no, stage order), "
                    "trained on APTOS 2019. Each image is checked 20 times (4 flips x Monte-Carlo dropout) "
                    f"to measure certainty.\n\nTest accuracy **{tm.get('acc', float('nan')):.3f}**, "
                    f"macro-F1 **{tm.get('macro_f1', float('nan')):.3f}**, QWK **{tm.get('qwk', float('nan')):.3f}**.")
    st.markdown(DISCLAIMER)

ex_dir = os.path.join(BASE_DIR, "examples")
examples = sorted(os.listdir(ex_dir)) if os.path.isdir(ex_dir) else []

col_in, col_out = st.columns([1, 1.3], gap="large")
with col_in:
    section(1, "Choose an image")
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
    section(2, "Result")
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
            k = r["stage"]
            color = STAGE_COLORS[k]
            review = "REVIEW" in r["decision"]
            dr_yes = r["dr_prob"] >= 0.5
            unsure = r["entropy"] > CFG["entropy_threshold"]
            # --- result card: stage, plain explanation and three small tiles ---
            st.markdown(f"""
<div class="card" style="border-left: 8px solid {color};">
  <div class="muted">Predicted stage</div>
  <div class="stage-name" style="color:{color};">{html.escape(CLASS_NAMES[k])}</div>
  <div>{STAGE_TEXT[k]}</div>
  <div class="tiles">
    <div class="tile"><div class="t">Signs of DR</div>
      <div class="v" style="color:{'#c62828' if dr_yes else '#2e7d32'};">{'Yes' if dr_yes else 'No'}</div>
      <div class="muted">{r['dr_prob']:.0%} likely</div></div>
    <div class="tile"><div class="t">Model confidence</div>
      <div class="v">{r['probs'][k]:.0%}</div><div class="muted">for this stage</div></div>
    <div class="tile"><div class="t">Uncertainty</div>
      <div class="v" style="color:{'#ef6c00' if unsure else '#2e7d32'};">{r['entropy']:.2f}</div>
      <div class="muted">0 = sure, 1 = unsure</div></div>
  </div>
</div>""", unsafe_allow_html=True)

            # --- suggested action, coloured by urgency ---
            urgent = "Refer (urgent)" in r["decision"]
            refer = "Refer" in r["decision"]
            a_col, a_bg = (("#c62828", "#fdecea") if urgent else
                           ("#ef6c00", "#fff4e5") if (refer or review) else ("#2e7d32", "#edf7ed"))
            action = html.escape(r["decision"].replace(" + HUMAN GRADER REVIEW", ""))
            extra = "<div><b>Please have this checked by a specialist.</b></div>" if review else ""
            reasons = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
            st.markdown(f"""
<div class="action" style="background:{a_bg}; border-color:{a_col};">
  <h4 style="color:{a_col};">Suggested action: {action}</h4>{extra}<ul>{reasons}</ul>
</div>""", unsafe_allow_html=True)
            if r["warnings"]:
                st.warning("Image quality: " + "; ".join(r["warnings"]))

            # --- probability of every stage as coloured bars ---
            bars = "".join(
                f'<div class="bar-row"><div class="bar-label">{"<b>" if i == k else ""}{i} - {html.escape(n)}'
                f'{"</b>" if i == k else ""}</div><div class="bar-track"><div class="bar-fill" '
                f'style="width:{p * 100:.1f}%; background:{STAGE_COLORS[i]};"></div></div>'
                f'<div class="bar-pct">{p:.0%}</div></div>'
                for i, (n, p) in enumerate(zip(CLASS_NAMES, r["probs"])))
            st.markdown(f'<div class="card"><b>Probability of each stage</b>{bars}</div>', unsafe_allow_html=True)
            st.download_button("Download screening report", make_report(r),
                               file_name=f"dr_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")

if image is not None and not r["rejected"]:
    with st.expander("See what the model looked at", expanded=True):
        c1, c2 = st.columns(2)
        c1.image(r["proc"], caption="Enhanced image used by the model", width="stretch")
        c2.image(r["overlay"], caption="Grad-CAM: red areas influenced the result most", width="stretch")

st.divider()
section(3, "Ask about your result")
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
