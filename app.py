import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import enhance_image, extract_text
import google.generativeai as genai
import os

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.warning("⚠️ Gemini API key not found. App will run in offline mode.")
else:
    genai.configure(api_key=API_KEY)

st.title("📝 Scribble to Digital")
st.write("Convert messy handwritten notes into clean text & to-do lists")

uploaded_file = st.file_uploader("Upload notes image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # Enhance image
    enhanced = enhance_image(img_array)
    st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    # OCR
    with st.spinner("🔍 Extracting text with OCR..."):
        raw_text = extract_text(enhanced)

    st.subheader("📄 Raw OCR Text")
    st.text(raw_text)

    if st.button("✨ Convert to Digital"):

        if raw_text.strip() == "":
            st.warning("⚠️ No text detected in the image.")
        else:

            with st.spinner("🤖 Processing..."):

                prompt = f"""
Clean this OCR text and extract to-do tasks.

OCR Text:
{raw_text}
"""

                try:
                    if API_KEY:
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        response = model.generate_content(prompt)
                        result = response.text
                    else:
                        raise Exception("Offline mode")

                except Exception:

                    # Offline fallback (no Gemini needed)
                    lines = raw_text.split("\n")
                    clean_lines = [line.strip() for line in lines if line.strip() != ""]

                    clean_notes = "\n".join([f"- {line}" for line in clean_lines])
                    todo_list = "\n".join([f"- {line}" for line in clean_lines[:3]])

                    result = f"""
Clean Notes:
{clean_notes}

To-Do List:
{todo_list}
"""

        st.subheader("✅ Digital Output")
        st.markdown(result)