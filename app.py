import os
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────────
# Works both locally and on Streamlit Cloud
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224

RECYCLING_TIPS = {
    "cardboard": "♻️ Flatten boxes before recycling. Keep dry — wet cardboard contaminates the whole bin.",
    "glass":     "🫙 Rinse jars and bottles. Remove lids. Don't mix with window glass or light bulbs.",
    "metal":     "🥫 Rinse cans. Aluminium cans can be recycled infinitely. Empty aerosols first.",
    "paper":     "📄 Keep paper dry. Shredded paper in a sealed bag. Glossy paper is recyclable.",
    "plastic":   "🧴 Check resin code (1–7). Types 1 (PET) and 2 (HDPE) most widely accepted.",
    "trash":     "🗑️ Doesn't fit a recyclable category. Consider reducing or reusing first.",
}

CLASS_COLOURS = {
    "cardboard": "#8D6E63", "glass": "#29B6F6", "metal": "#78909C",
    "paper": "#FFA726",     "plastic": "#66BB6A", "trash": "#EF5350",
}

CLASS_ICONS = {
    "cardboard": "📦", "glass": "🫙", "metal": "🥫",
    "paper": "📄",     "plastic": "🧴", "trash": "🗑️",
}

st.set_page_config(page_title="Waste Classifier", page_icon="♻️",
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    .block-container {padding-top:2rem;}
    .stButton > button {
        width:100%; background:#4CAF50; color:white; border:none;
        padding:0.6rem; border-radius:8px; font-size:1rem; font-weight:600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    return tf.keras.models.load_model(path)


def predict(model, img):
    img  = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    top   = int(np.argmax(probs))
    return CLASSES[top], float(probs[top]), probs


def confidence_chart(probs):
    sorted_pairs = sorted(zip(CLASSES, probs), key=lambda x: x[1])
    cls_s  = [p[0] for p in sorted_pairs]
    prob_s = [p[1] for p in sorted_pairs]
    cols   = [CLASS_COLOURS[c] for c in cls_s]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    bars = ax.barh(cls_s, [p*100 for p in prob_s], color=cols, edgecolor="none", height=0.55)
    for bar, p in zip(bars, prob_s):
        ax.text(bar.get_width()+0.8, bar.get_y()+bar.get_height()/2,
                f"{p*100:.1f}%", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Confidence (%)", color="#aaa", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#333", linewidth=0.5)
    fig.tight_layout(pad=1.0)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    available = {}
    for name, fname in [("MobileNetV2 (recommended)", "mobilenetv2.h5"),
                        ("Custom CNN", "custom_cnn.h5")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            available[name] = path

    # Debug — show what files exist in models/
    if not available:
        st.error("❌ No models found.")
        files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        st.write("Files in models/:", files)
        st.write("Looking in:", MODEL_DIR)
        st.stop()

    choice     = st.selectbox("Model", list(available.keys()))
    model_path = available[choice]

    st.markdown("---")
    st.markdown("**Categories**")
    for cls in CLASSES:
        st.markdown(
            f'<span style="color:{CLASS_COLOURS[cls]}">⬤</span> '
            f'{CLASS_ICONS[cls]} {cls.capitalize()}',
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Dataset:** TrashNet · 2,527 images · 6 classes")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# ♻️ Waste Classifier")
st.markdown("Upload a photo of any waste item to classify it and get a recycling tip.")
st.markdown("---")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"],
                             label_visibility="collapsed")

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_column_width=True, caption="Uploaded image")

    with col2:
        model = load_model(model_path)
        with st.spinner("Classifying…"):
            label, confidence, probs = predict(model, img)

        colour = CLASS_COLOURS[label]
        icon   = CLASS_ICONS[label]
        st.markdown(f"""
            <div style="background:{colour}22;border:2px solid {colour};
                border-radius:12px;padding:20px;text-align:center;margin-bottom:12px;">
                <div style="font-size:2.8em;">{icon}</div>
                <div style="font-size:1.9em;font-weight:800;color:{colour};
                    letter-spacing:1px;margin:6px 0;">{label.upper()}</div>
                <div style="font-size:1.15em;color:#ddd;">{confidence*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

        if confidence < 0.50:
            st.warning("Low confidence — try a clearer photo.")
        elif confidence >= 0.85:
            st.success("High confidence prediction!")

    st.markdown("### 📊 Class Probabilities")
    fig = confidence_chart(probs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### ♻️ Recycling Guide")
    st.info(RECYCLING_TIPS[label])

    with st.expander("🔢 All scores"):
        for cls, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
            st.write(f"**{CLASS_ICONS[cls]} {cls.capitalize()}** — {p*100:.2f}%")

else:
    st.markdown("""
        <div style="border:2px dashed #444;border-radius:14px;padding:60px 20px;
            text-align:center;color:#666;margin-top:10px;">
            <div style="font-size:3em;">📸</div>
            <div style="font-size:1.1em;margin-top:12px;color:#888;">
                Drop an image here or click Browse</div>
            <div style="font-size:0.85em;margin-top:8px;color:#555;">
                Try: a plastic bottle, tin can, cardboard box, glass jar</div>
        </div>""", unsafe_allow_html=True)