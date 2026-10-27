import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.getenv("PPE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="PPE Detection API Demo", page_icon="🦺", layout="wide")
st.title("PPE Detection — API Demo")
st.caption("Streamlit is the client. FastAPI runs YOLO inference and returns structured JSON.")

with st.sidebar:
    st.header("API settings")
    api_url = st.text_input("FastAPI base URL", value=API_URL).rstrip("/")
    st.write("API docs:", f"{api_url}/docs")
    st.write("Supported model classes come from the trained weights.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded image")
    st.image(original, use_container_width=True)

    if st.button("Run detection", type="primary"):
        try:
            with st.spinner("Sending image to FastAPI..."):
                response = requests.post(
                    f"{api_url}/detect/image",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
                    timeout=120,
                )
            if response.status_code != 200:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail}")
            else:
                payload = response.json()
                annotated = original.copy()
                draw = ImageDraw.Draw(annotated)
                for item in payload.get("detections", []):
                    x1, y1, x2, y2 = item["bbox"]
                    label = f'{item["class_name"]} {item["confidence"]:.2f}'
                    draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
                    text_box = draw.textbbox((x1, max(0, y1 - 18)), label)
                    draw.rectangle(text_box, fill="lime")
                    draw.text((x1, max(0, y1 - 18)), label, fill="black")

                left, right = st.columns(2)
                with left:
                    st.subheader("Annotated result")
                    st.image(annotated, use_container_width=True)
                    out = io.BytesIO()
                    annotated.save(out, format="PNG")
                    st.download_button("Download annotated image", out.getvalue(),
                                       file_name="ppe_detection_result.png", mime="image/png")
                with right:
                    st.subheader(f'Detections ({payload["count"]})')
                    detections = payload.get("detections", [])
                    if detections:
                        table = pd.DataFrame([
                            {
                                "Class": d["class_name"],
                                "Confidence": d["confidence"],
                                "X1": d["bbox"][0], "Y1": d["bbox"][1],
                                "X2": d["bbox"][2], "Y2": d["bbox"][3],
                            } for d in detections
                        ])
                        st.dataframe(table, use_container_width=True)
                        st.download_button("Download detections CSV", table.to_csv(index=False),
                                           file_name="detections.csv", mime="text/csv")
                    else:
                        st.info("No objects were detected above the model's default confidence threshold.")
                with st.expander("Raw JSON response"):
                    st.json(payload)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Start the backend on port 8000 first.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
else:
    st.info("Upload an image to send it to the FastAPI detection endpoint.")
