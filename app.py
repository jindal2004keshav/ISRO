import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
import cv2

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

st.title("YOLOv5 Object Detection with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Perform detection
    results = model(img_array)
    
    # Display results
    st.write("Detection Results:")
    st.dataframe(results.pandas().xyxy[0])

    # Draw bounding boxes on the image
    img_with_boxes = img_array.copy()
    for _, row in results.pandas().xyxy[0].iterrows():
        cv2.rectangle(img_with_boxes, 
                      (int(row['xmin']), int(row['ymin'])), 
                      (int(row['xmax']), int(row['ymax'])), 
                      (255, 0, 0), 2)
    
    # Convert to PIL image for display
    img_with_boxes_pil = Image.fromarray(img_with_boxes)
    st.image(img_with_boxes_pil, caption='Image with Detections', use_column_width=True)
