# app/app.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from infer import get_model, list_checkpoints, DEFAULT_MODEL_LOC, DEVICE

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠")

st.title("🧠 Sentiment Analysis (DistilBERT)")
st.info("Enter text below and click **Analyze** to classify sentiment.")

# ----------------------- Checkpoint discovery -----------------------
env_path = os.getenv("MODEL_PATH")
model_dir = (
    Path(env_path).parent
    if (env_path and Path(env_path).is_file())
    else Path(env_path or DEFAULT_MODEL_LOC)
)

if env_path and Path(env_path).is_file():
    chosen_ckpt = Path(env_path)
else:
    ckpts = list_checkpoints(model_dir)
    if not ckpts:
        st.error(
            f"No .pth checkpoints found in `{model_dir}`.\n"
            "Place files under `saved_models/` or set `MODEL_PATH` to a specific file."
        )
        st.stop()
    names = [p.name for p in ckpts]
    sel = st.sidebar.selectbox("Checkpoint", names, index=0)
    chosen_ckpt = next(p for p in ckpts if p.name == sel)

with st.expander("Runtime & Model", expanded=False):
    st.write(f"**Device**: `{DEVICE}`")
    st.write(f"**Checkpoint**: `{chosen_ckpt.name}`")
    st.write(f"**Model directory**: `{model_dir}`")
    if env_path:
        st.write(f"**MODEL_PATH (env)**: `{env_path}`")


# ----------------------- Examples (optional ground truth) -----------------------
examples = [
    ("I absolutely love this phone, the battery lasts forever!", "Positive comment"),
    ("This is the worst purchase I have ever made.", "Negative comment"),
    ("The strong point of this novel is the humor employed by Neil Gaiman.", "Positive comment"),
]
ex_options = ["— Select example —"] + [e[0] for e in examples]
picked = st.sidebar.selectbox("Quick examples", ex_options, index=0)

if picked != ex_options[0]:
    idx = ex_options.index(picked) - 1
    st.session_state["input_text"] = examples[idx][0]
    st.session_state["actual_label"] = examples[idx][1]
else:
    st.session_state.pop("actual_label", None)  # clear only the example-derived label

# Show a short preview of the selected example above the input.
sel = st.session_state.get("input_text", "")
if sel:
    st.caption(f"Selected example → {sel[:120]}{'…' if len(sel) > 120 else ''}")


# ----------------------- SAFE, NO-CSS INPUT BLOCK -----------------------
st.markdown("### Input")

# Let users switch between a multi-line textarea and a single-line input if needed.
mode = st.selectbox("Input mode", ["Textarea (multi-line)", "Textbox (single line)"], index=0)

if mode.startswith("Textarea"):
    text = st.text_area(
        "Enter text",
        value=st.session_state.get("input_text", ""),
        height=160,
        key="user_textarea_safe",  # unique key avoids widget-state collisions
        label_visibility="visible",
        placeholder="Type a review here or pick an example from the sidebar…",
    )
else:
    text = st.text_input(
        "Enter text",
        value=st.session_state.get("input_text", ""),
        key="user_textbox_safe",  # separate key for textbox path
        label_visibility="visible",
        placeholder="Type a review here or pick an example from the sidebar…",
    )

# Simple, robust controls.
c1, c2 = st.columns([2, 1])
with c1:
    max_len = st.slider("Max sequence length", 32, 512, 256, 32, key="len_slider_safe")
with c2:
    show_probs = st.toggle("Show probabilities", value=True, key="probs_toggle_safe")


# ----------------------- Inference -----------------------
@st.cache_resource(show_spinner=True)
def load_cached_model(path_str: str):
    """Cache the heavy PyTorch model per checkpoint path."""
    return get_model(path_str)

def _split_lines(s: str) -> list[str]:
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines if len(lines) > 1 else ([s.strip()] if s.strip() else [])

if st.button("Analyze", type="primary", key="analyze_btn_safe"):
    texts = _split_lines(text)
    if not texts:
        st.warning("Please enter a longer text.")
        st.stop()

    try:
        sm = load_cached_model(str(chosen_ckpt))
    except Exception as e:
        st.error(f"Model could not be loaded.\n\n{e}")
        st.stop()

    preds, probs = sm.predict(texts, max_len=max_len)
    id2label = sm.id2label

    # Build a minimal results table: Actual, Predicted, Probability.
    actual_label = st.session_state.get("actual_label")
    rows = []
    for i in range(len(texts)):
        pred_idx = int(preds[i])
        pred_label = id2label[pred_idx]
        prob_pred = float(np.max(probs[i]))  # probability of the predicted class
        rows.append(
            {
                "Actual": actual_label if actual_label else "—",
                "Predicted": pred_label,
                "Probability": round(prob_pred, 4),
            }
        )

    st.markdown("### Results")
    st.dataframe(pd.DataFrame(rows, columns=["Actual", "Predicted", "Probability"]),
                 use_container_width=True)