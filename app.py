# ============================================================
# CROP DISEASE DETECTION - STREAMLIT WEB APP
# File: app.py
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# DISEASE REMEDIES — Matched exactly to your 15 classes
# -------------------------------------------------------
REMEDIES = {
    # ── PEPPER ──────────────────────────────────────────
    "Pepper__bell___Bacterial_spot": {
        "display": "Pepper — Bacterial Spot",
        "remedy": (
            "Apply copper-based bactericides (e.g., Kocide 3000) every 7–10 days. "
            "Remove and destroy heavily infected leaves immediately. "
            "Avoid overhead irrigation — water at the base of the plant. "
            "Use disease-free certified seeds for the next season. "
            "Rotate crops — do not plant peppers or tomatoes in the same spot next year."
        ),
        "severity": "High"
    },
    "Pepper__bell___healthy": {
        "display": "Pepper — Healthy ✅",
        "remedy": (
            "Your pepper plant looks perfectly healthy! "
            "Continue regular watering at the base, fertilize every 3–4 weeks with a balanced NPK fertilizer, "
            "and monitor for pests like aphids or whiteflies. "
            "Ensure good air circulation between plants."
        ),
        "severity": "None"
    },

    # ── POTATO ──────────────────────────────────────────
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "remedy": (
            "Apply fungicides containing chlorothalonil or mancozeb every 7–14 days. "
            "Remove infected lower leaves and dispose of them away from the field — do not compost. "
            "Avoid wetting the foliage — use drip irrigation if possible. "
            "Rotate crops: avoid planting potatoes or tomatoes in the same soil for 2–3 years. "
            "Ensure proper spacing to allow airflow between plants."
        ),
        "severity": "Medium"
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight 🚨",
        "remedy": (
            "URGENT ACTION REQUIRED — Late blight spreads extremely rapidly and can destroy an entire crop in days. "
            "Immediately apply fungicides containing metalaxyl, cymoxanil, or mancozeb. "
            "Remove and destroy ALL infected plant material — burn or bury it, do NOT compost. "
            "Harvest tubers as soon as possible if infection is severe. "
            "Do not plant potatoes in the same field for at least 3 years. "
            "Consult your local agriculture officer immediately."
        ),
        "severity": "Severe"
    },
    "Potato___healthy": {
        "display": "Potato — Healthy ✅",
        "remedy": (
            "Your potato plant is healthy! "
            "Continue regular hilling (mounding soil around the base) to protect tubers from sunlight. "
            "Water consistently but avoid waterlogging. "
            "Monitor for Colorado potato beetle and aphids. "
            "Fertilize with potassium-rich fertilizer to support tuber development."
        ),
        "severity": "None"
    },

    # ── TOMATO ──────────────────────────────────────────
    "Tomato_Bacterial_spot": {
        "display": "Tomato — Bacterial Spot",
        "remedy": (
            "Apply copper-based bactericides (copper hydroxide or copper sulfate) at first sign of infection. "
            "Remove and destroy infected leaves. "
            "Avoid overhead watering — water at soil level in the morning. "
            "Disinfect all garden tools with 10% bleach solution. "
            "Use certified disease-free transplants next season and practice 2-year crop rotation."
        ),
        "severity": "High"
    },
    "Tomato_Early_blight": {
        "display": "Tomato — Early Blight",
        "remedy": (
            "Apply fungicides (chlorothalonil, mancozeb, or copper-based) every 7–10 days. "
            "Remove affected lower leaves — this is where the disease starts. "
            "Mulch around the base to prevent soil splash onto leaves. "
            "Stake or cage plants to improve airflow. "
            "Avoid watering in the evening — wet leaves overnight encourage disease spread."
        ),
        "severity": "Medium"
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight 🚨",
        "remedy": (
            "URGENT — Late blight can destroy your entire crop within a week if not treated immediately. "
            "Apply fungicides containing metalaxyl or cymoxanil right away. "
            "Remove and destroy all infected plant parts — burn or bury them, do NOT compost. "
            "Avoid working in the field when plants are wet to prevent further spread. "
            "Harvest any unaffected tomatoes immediately and monitor stored tomatoes daily."
        ),
        "severity": "Severe"
    },
    "Tomato_Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "remedy": (
            "Apply fungicides containing chlorothalonil or copper-based products. "
            "Improve ventilation in greenhouse or polytunnel — open vents and increase spacing. "
            "Reduce humidity below 85% — leaf mold thrives in high humidity. "
            "Remove and destroy infected leaves. "
            "Avoid wetting the foliage during irrigation. "
            "Use mold-resistant tomato varieties in the next growing season."
        ),
        "severity": "Medium"
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "remedy": (
            "Apply fungicides (mancozeb, chlorothalonil, or copper) every 7–10 days. "
            "Remove and discard all infected leaves immediately — this disease spreads through spores on fallen leaves. "
            "Mulch around the base of plants to prevent soil splash. "
            "Avoid overhead irrigation. "
            "Rotate crops for 2 years — Septoria survives in soil and on plant debris."
        ),
        "severity": "Medium"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Tomato — Spider Mites (Two-Spotted)",
        "remedy": (
            "Spider mites thrive in hot, dry conditions — increase humidity around plants by misting. "
            "Apply miticides (abamectin or bifenazate) or insecticidal soap spray directly on the underside of leaves. "
            "Introduce natural predators: predatory mites (Phytoseiulus persimilis) are very effective. "
            "Remove heavily infested leaves immediately. "
            "Avoid using broad-spectrum pesticides which kill natural predators of mites."
        ),
        "severity": "Medium"
    },
    "Tomato__Target_Spot": {
        "display": "Tomato — Target Spot",
        "remedy": (
            "Apply fungicides containing azoxystrobin, chlorothalonil, or mancozeb at first sign of disease. "
            "Remove infected leaves from the lower canopy. "
            "Improve plant spacing to increase airflow — this disease spreads in humid, stagnant air. "
            "Avoid overhead irrigation. "
            "Practice crop rotation — do not plant tomatoes in the same soil for 2 seasons."
        ),
        "severity": "Medium"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus 🚨",
        "remedy": (
            "There is NO cure for Yellow Leaf Curl Virus — infected plants must be removed and destroyed. "
            "The virus is spread by whiteflies — control whitefly populations immediately using imidacloprid or spinosad. "
            "Use reflective silver mulch to repel whiteflies. "
            "Install yellow sticky traps to monitor whitefly levels. "
            "Plant only virus-resistant tomato varieties (labelled TYLCV-resistant) in the next season. "
            "Inspect plants daily and remove any newly infected plants immediately to protect remaining healthy plants."
        ),
        "severity": "Severe"
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "remedy": (
            "There is NO cure — remove and destroy infected plants immediately to protect others. "
            "The virus spreads through touch, tools, and sap — wash hands thoroughly before handling plants. "
            "Disinfect all tools with 10% bleach or 70% alcohol between plants. "
            "Control aphids (the main vector) with insecticidal soap or neem oil. "
            "Do not smoke near tomato plants — tobacco can carry a related mosaic virus. "
            "Plant mosaic-resistant varieties in the next season."
        ),
        "severity": "High"
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy ✅",
        "remedy": (
            "Your tomato plant is in great health! "
            "Continue consistent deep watering (avoid wetting leaves), fertilize with calcium-rich fertilizer to prevent blossom end rot, "
            "and stake or cage the plant for support. "
            "Check the undersides of leaves weekly for pests like aphids or spider mites. "
            "Remove suckers (side shoots) to improve airflow and fruit size."
        ),
        "severity": "None"
    },
}

SEVERITY_COLOR = {
    "None":   "#2d6a4f",
    "Medium": "#e07b00",
    "High":   "#cc3300",
    "Severe": "#990000"
}

SEVERITY_EMOJI = {
    "None":   "✅",
    "Medium": "⚠️",
    "High":   "🔴",
    "Severe": "🚨"
}

# -------------------------------------------------------
# LOAD MODEL (cached so it only loads once)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "crop_disease_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'crop_disease_model.h5' not found! Make sure it is in the same folder as app.py.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_class_names():
    if not os.path.exists("class_names.json"):
        st.error("❌ 'class_names.json' not found!")
        return []
    with open("class_names.json", "r") as f:
        return json.load(f)

# -------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------
def predict(model, class_names, image: Image.Image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    top3_indices = np.argsort(predictions)[::-1][:3]
    results = []
    for idx in top3_indices:
        raw_name = class_names[idx]
        info = REMEDIES.get(raw_name, {})
        results.append({
            "class":      raw_name,
            "display":    info.get("display", raw_name.replace("_", " ")),
            "confidence": float(predictions[idx]) * 100,
            "remedy":     info.get("remedy", "Consult a local agricultural expert."),
            "severity":   info.get("severity", "Unknown")
        })
    return results

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#2d6a4f;'>🌿 Crop Disease Detector</h1>
<p style='text-align:center; color:#555; font-size:17px;'>
Upload a photo of a crop leaf to instantly detect diseases and get treatment advice.
</p>
<hr style='border:1px solid #d4edda; margin-bottom:20px;'>
""", unsafe_allow_html=True)

# Supported crops info box
with st.expander("📋 Supported Crops & Diseases (15 classes)"):
    st.markdown("""
    | Crop | Conditions Detected |
    |------|-------------------|
    | 🌶️ **Bell Pepper** | Bacterial Spot, Healthy |
    | 🥔 **Potato** | Early Blight, Late Blight, Healthy |
    | 🍅 **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

    **Tips for best results:**
    - Use a clear, well-lit photo of a single leaf
    - Make sure the leaf fills most of the frame
    - Avoid blurry or dark images
    """)

# Upload
st.markdown("### 📷 Upload a Leaf Photo")
uploaded_file = st.file_uploader(
    "Choose an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        st.markdown("### 🔬 Analysis Result")
        with st.spinner("Analyzing... please wait"):
            model       = load_model()
            class_names = load_class_names()

            if model and class_names:
                results    = predict(model, class_names, image)
                top        = results[0]
                severity   = top["severity"]
                sev_color  = SEVERITY_COLOR.get(severity, "#555")
                sev_emoji  = SEVERITY_EMOJI.get(severity, "")

                # Result box
                if severity == "None":
                    st.success(f"**{top['display']}**")
                elif severity == "Severe":
                    st.error(f"**{top['display']}**")
                else:
                    st.warning(f"**{top['display']}**")

                # Confidence
                st.markdown(f"**Confidence:** {top['confidence']:.1f}%")
                st.progress(min(int(top['confidence']), 100))

                # Severity
                st.markdown(
                    f"**Severity:** <span style='color:{sev_color}; font-weight:bold;'>"
                    f"{sev_emoji} {severity}</span>",
                    unsafe_allow_html=True
                )

    # Treatment — full width
    st.markdown("---")
    st.markdown("### 💊 Recommended Treatment")
    if top["severity"] == "None":
        st.success(f"🎉 **{top['display']}**\n\n{top['remedy']}")
    elif top["severity"] == "Severe":
        st.error(f"🚨 **{top['display']}**\n\n{top['remedy']}")
    elif top["severity"] == "High":
        st.warning(f"🔴 **{top['display']}**\n\n{top['remedy']}")
    else:
        st.info(f"⚠️ **{top['display']}**\n\n{top['remedy']}")

    # Other possibilities
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### 📊 Other Possibilities")
        for i, res in enumerate(results[1:], 2):
            st.markdown(f"**#{i}** {res['display']} — `{res['confidence']:.1f}%` confidence")

    st.markdown("---")
    st.info(
        "ℹ️ **Disclaimer:** This AI tool is designed to assist farmers and is not a replacement "
        "for professional agricultural advice. For severe cases, please consult a certified "
        "plant pathologist or your local agriculture department."
    )

else:
    # Placeholder
    st.markdown("""
    <div style='
        border: 2px dashed #2d6a4f;
        border-radius: 12px;
        padding: 60px 20px;
        text-align: center;
        color: #555;
        background-color: #f0f9f4;
        margin-top: 20px;
    '>
        <h3 style="color:#2d6a4f;">📸 Upload a leaf photo to get started</h3>
        <p>Supports Bell Pepper, Potato, and Tomato leaves</p>
        <p style="font-size:13px; color:#888;">Accepted formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:40px;">
<p style='text-align:center; color:#aaa; font-size:12px;'>
🌿 Crop Disease Detector | Microsoft Elevate AICTE Internship Project<br>
Built with TensorFlow MobileNetV2 + Streamlit | Accuracy: 95.05%
</p>
""", unsafe_allow_html=True)
