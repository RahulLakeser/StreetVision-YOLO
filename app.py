
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
from io import BytesIO
import pandas as pd
import time

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="StreetVision AI",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 StreetVision AI")
st.write(
    "Upload a street image to detect vehicles, people, "
    "and other objects using YOLO."
)

# --------------------------------------------------
# 2. Load YOLO Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    
    # Compare both models on your own images.
    return YOLO("yolov8m.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the YOLO model: {e}")
    st.stop()

# --------------------------------------------------
# 3. Sidebar Settings
# --------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help=(
        "Detections below this confidence threshold "
        "will be filtered out."
    )
)

# COCO classes supported by the pretrained model.
available_classes = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic light": 9,
    "stop sign": 11
}

selected_objects = st.sidebar.multiselect(
    "Objects to Detect",
    options=list(available_classes.keys()),
    default=[
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck"
    ]
)

st.sidebar.caption(
    "Note: Detection quality depends on the model, "
    "image quality, object size, and visibility."
)

# --------------------------------------------------
# 4. Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a street image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to read this image. Please upload a valid image.")
        st.stop()

    st.subheader("Original Image")

    st.image(
        image,
        caption="Uploaded street image",
        use_container_width=True
    )

    if not selected_objects:
        st.warning("Select at least one object category in the sidebar.")
        st.stop()

    if st.button("🔍 Detect Objects", type="primary"):

        selected_class_ids = [
            available_classes[name]
            for name in selected_objects
        ]

        with st.spinner("Detecting objects..."):

            start_time = time.perf_counter()
            
            results = model.predict(
                source=image,
                device="cpu",
                conf=confidence,
                imgsz=960,
                iou=0.65,
                verbose=False
            )

            elapsed = time.perf_counter() - start_time

        result = results[0]

        # --------------------------------------------------
        # 5. Annotated Image
        # --------------------------------------------------

        annotated_array = result.plot()[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_array)

        st.subheader("🎯 Detection Results")

        st.image(
            annotated_image,
            caption="Detected objects with bounding boxes",
            use_container_width=True
        )

        # --------------------------------------------------
        # 6. Extract Detection Information
        # --------------------------------------------------

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = (
                round(float(value), 2)
                for value in box.xyxy[0]
            )

            detections.append({
                "Object": class_name,
                "Confidence (%)": round(score * 100, 2),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })

        # --------------------------------------------------
        # 7. Summary Metrics
        # --------------------------------------------------

        counts = Counter(
            item["Object"]
            for item in detections
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Objects Detected",
            len(detections)
        )

        col2.metric(
            "Object Categories",
            len(counts)
        )

        col3.metric(
            "Inference Time",
            f"{elapsed * 1000:.1f} ms"
        )

        # --------------------------------------------------
        # 8. Detection Summary
        # --------------------------------------------------

        st.subheader("📊 Detection Summary")

        if detections:

            df = pd.DataFrame(detections)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Objects by Category")

            category_counts = pd.DataFrame(
                counts.items(),
                columns=["Object", "Count"]
            ).set_index("Object")

            st.bar_chart(category_counts)

            # --------------------------------------------------
            # 9. Download Annotated Image
            # --------------------------------------------------

            image_buffer = BytesIO()
            annotated_image.save(image_buffer, format="PNG")

            st.download_button(
                label="⬇️ Download Annotated Image",
                data=image_buffer.getvalue(),
                file_name="streetvision_result.png",
                mime="image/png"
            )

            # --------------------------------------------------
            # 10. Download Detection Data
            # --------------------------------------------------

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Detection Data (CSV)",
                data=csv_data,
                file_name="streetvision_detections.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "No objects were detected at this confidence "
                "threshold. Try lowering the threshold or "
                "uploading another image."
            )
