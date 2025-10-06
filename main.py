import streamlit as st
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Chest X-ray Pneumonia Detection", layout="centered")

# --- مسار النموذج داخل الريبو ---
MODEL_PATH = "pneu_d_models2.h5"  # ضع الملف في نفس مجلد main.py

# --- تحميل النموذج مع cache ---
@st.cache_resource
def load_pneu_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ لم يتم العثور على النموذج: {MODEL_PATH}")
        return None
    model = load_model(MODEL_PATH)
    return model

model = load_pneu_model()

if model is None:
    st.stop()  # إذا النموذج غير موجود، أوقف التطبيق

# --- استخراج شكل الإدخال من النموذج ---
input_shape = model.input_shape[1:3]   # (height, width)
input_channels = model.input_shape[3]  # 1=grayscale, 3=rgb

# --- واجهة التطبيق ---
st.title("🩻 Chest X-ray Pneumonia Detection")
st.write("ارفع صورة أشعة للصدر وسيتم التنبؤ إذا كانت **طبيعية** أو **التهاب رئوي**")

# --- رفع الصورة ---
uploaded_file = st.file_uploader("ارفع صورة X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- فتح الصورة حسب عدد القنوات ---
    if input_channels == 1:
        img = Image.open(uploaded_file).convert("L")
    else:
        img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="📷 الصورة المرفوعة", use_column_width=True)

    # --- تجهيز الصورة للتنبؤ ---
    img = img.resize(input_shape)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- شريط تحميل للتنبؤ ---
    progress_bar = st.progress(0)
    for i in range(1, 101):
        time.sleep(0.005)
        progress_bar.progress(i)

    # --- التنبؤ ---
    prediction = model.predict(img_array)[0][0]

    pneumonia_prob = float(prediction * 100)
    normal_prob = float((1 - prediction) * 100)

    # --- عرض النتائج ---
    st.subheader("🔍 النتيجة:")
    if prediction > 0.5:
        st.error(f"📌 الحالة: Pneumonia (التهاب رئوي)\n\n🔴 احتمال الالتهاب: {pneumonia_prob:.2f}%")
    else:
        st.success(f"📌 الحالة: Normal (طبيعي)\n\n🟢 احتمال الطبيعي: {normal_prob:.2f}%")

    # --- جدول الاحتمالات ---
    probs_df = pd.DataFrame({
        "التصنيف": ["Normal (طبيعي)", "Pneumonia (التهاب رئوي)"],
        "الاحتمال %": [normal_prob, pneumonia_prob]
    })
    st.table(probs_df)

# --- فوتر منسّق ---
st.markdown(
    """
    <hr style='border:1px solid gray; margin-top:40px; margin-bottom:10px;'>
    <p style='text-align:center; font-weight:bold;'>
        تم التطوير بواسطة <span style='color:#2E86C1;'>م. منير البحيري </span> 🔬<br>
        © جميع الحقوق محفوظة 2025
    </p>
    """,
    unsafe_allow_html=True
)
