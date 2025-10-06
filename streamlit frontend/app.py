import streamlit as st
import requests
from io import BytesIO
from PIL import Image

# 🎯 FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000"

# 🩺 App title and description
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2b6cb0;'>🩻 Pneumonia Detection App</h1>
    <p style='text-align: center; font-size: 18px; color: #555;'>
        Upload a chest X-ray image and let the AI model analyze it for signs of pneumonia.
    </p>
    """,
    unsafe_allow_html=True
)

# 📤 File upload
uploaded_file = st.file_uploader("📎 Upload chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image... Please wait..."):
        try:
            # Send file to FastAPI backend
            files = {"file": uploaded_file.getbuffer()}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)

            if response.status_code == 200:
                data = response.json()
                prediction = data.get("Prediction", "Unknown")
                probability = data.get("Probability", 0)
                confidence = f"{probability * 100:.2f}%"

                # 🧠 Display results in styled boxes
                if prediction.lower() == "pneumonia":
                    color = "#ff4d4d"
                    emoji = "⚠️"
                else:
                    color = "#4CAF50"
                    emoji = "✅"

                st.markdown(
                    f"""
                    <div style='text-align: center; background-color: {color}; padding: 25px;
                                border-radius: 15px; color: white; margin-top: 20px;'>
                        <h2 style='margin: 0;'>{emoji} Prediction: {prediction}</h2>
                        <h4 style='margin-top: 10px;'>Confidence: {confidence}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.error(f"🚨 Server Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Error — Make sure your FastAPI server is running.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

# Footer
st.markdown(
    """
    <hr style='margin-top: 50px;'>
    <p style='text-align: center; color: #888; font-size: 14px;'>
        Built with ❤️ using Streamlit & FastAPI
    </p>
    """,
    unsafe_allow_html=True
)
