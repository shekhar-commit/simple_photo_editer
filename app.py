import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Photo Editor", page_icon="📷", layout="wide")
st.title("📷 Photo Editor")
st.write("Upload an image and apply various edits using OpenCV!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is not None:
        # Initialize session state for current image if not already done
        if 'current_image' not in st.session_state:
            st.session_state.current_image = image.copy()
        
        st.sidebar.header("Edit Options")
        operation = st.sidebar.selectbox(
            "Choose an operation",
            [
                "None",
                "Grayscale",
                "Resize",
                "Crop",
                "Flip",
                "Draw Line",
                "Draw Rectangle",
                "Draw Circle",
                "Add Text",
                "Gaussian Blur",
                "Median Blur",
                "Sharpen",
                "Canny Edge Detection",
                "Threshold",
                "Rotate",
                "Adjust Brightness/Contrast",
                "Color Adjustment (Hue/Saturation)",
            ]
        )
        
        # Parameters (same as before)
        if operation == "Resize":
            width = st.sidebar.slider("Width", 50, 1000, 300)
            height = st.sidebar.slider("Height", 50, 1000, 300)
        elif operation == "Crop":
            x1 = st.sidebar.slider("Start X", 0, image.shape[1]-1, 100)
            y1 = st.sidebar.slider("Start Y", 0, image.shape[0]-1, 100)
            x2 = st.sidebar.slider("End X", x1+1, image.shape[1], 200)
            y2 = st.sidebar.slider("End Y", y1+1, image.shape[0], 200)
        elif operation == "Draw Line":
            start_x = st.sidebar.slider("Start X", 0, image.shape[1], 100)
            start_y = st.sidebar.slider("Start Y", 0, image.shape[0], 200)
            end_x = st.sidebar.slider("End X", 0, image.shape[1], 150)
            end_y = st.sidebar.slider("End Y", 0, image.shape[0], 350)
            color = st.sidebar.color_picker("Color", "#FF0000")
            thickness = st.sidebar.slider("Thickness", 1, 10, 4)
        elif operation == "Draw Rectangle":
            start_x = st.sidebar.slider("Start X", 0, image.shape[1], 100)
            start_y = st.sidebar.slider("Start Y", 0, image.shape[0], 200)
            end_x = st.sidebar.slider("End X", 0, image.shape[1], 150)
            end_y = st.sidebar.slider("End Y", 0, image.shape[0], 350)
            color = st.sidebar.color_picker("Color", "#FF0000")
            thickness = st.sidebar.slider("Thickness", 1, 10, 4)
        elif operation == "Draw Circle":
            center_x = st.sidebar.slider("Center X", 0, image.shape[1], 300)
            center_y = st.sidebar.slider("Center Y", 0, image.shape[0], 300)
            radius = st.sidebar.slider("Radius", 10, 300, 150)
            color = st.sidebar.color_picker("Color", "#FF0000")
            thickness = st.sidebar.slider("Thickness", 1, 10, 5)
        elif operation == "Add Text":
            text = st.sidebar.text_input("Text", "This is my image")
            pos_x = st.sidebar.slider("Position X", 0, image.shape[1], 200)
            pos_y = st.sidebar.slider("Position Y", 0, image.shape[0], 300)
            font_scale = st.sidebar.slider("Font Scale", 0.5, 5.0, 1.0)
            color = st.sidebar.color_picker("Color", "#FF0000")
            thickness = st.sidebar.slider("Thickness", 1, 10, 2)
        elif operation == "Gaussian Blur":
            kernel_size = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
        elif operation == "Median Blur":
            kernel_size = st.sidebar.slider("Kernel Size", 1, 31, 1, step=2)
        elif operation == "Sharpen":
            pass  # No params needed
        elif operation == "Canny Edge Detection":
            min_thresh = st.sidebar.slider("Min Threshold", 0, 255, 50)
            max_thresh = st.sidebar.slider("Max Threshold", 0, 255, 100)
        elif operation == "Threshold":
            thresh_value = st.sidebar.slider("Threshold Value", 0, 255, 120)
        elif operation == "Rotate":
            angle = st.sidebar.slider("Angle", -180, 180, 0)
        elif operation == "Adjust Brightness/Contrast":
            brightness = st.sidebar.slider("Brightness", -100, 100, 0)
            contrast = st.sidebar.slider("Contrast", 0.0, 3.0, 1.0)
        elif operation == "Color Adjustment (Hue/Saturation)":
            hue = st.sidebar.slider("Hue", -180, 180, 0)
            saturation = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0)
        
        # Apply operation and update current image
        if operation != "None":
            processed_image = st.session_state.current_image.copy()
            
            if operation == "Grayscale":
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            elif operation == "Resize":
                processed_image = cv2.resize(processed_image, (width, height))
            elif operation == "Crop":
                processed_image = processed_image[y1:y2, x1:x2]
            elif operation == "Flip":
                processed_image = cv2.flip(processed_image, 1)
            elif operation == "Draw Line":
                color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                cv2.line(processed_image, (start_x, start_y), (end_x, end_y), color_bgr, thickness)
            elif operation == "Draw Rectangle":
                color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                cv2.rectangle(processed_image, (start_x, start_y), (end_x, end_y), color_bgr, thickness)
            elif operation == "Draw Circle":
                color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                cv2.circle(processed_image, (center_x, center_y), radius, color_bgr, thickness)
            elif operation == "Add Text":
                color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                cv2.putText(processed_image, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness)
            elif operation == "Gaussian Blur":
                processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), 1)
            elif operation == "Median Blur":
                processed_image = cv2.medianBlur(processed_image, kernel_size)
            elif operation == "Sharpen":
                sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                processed_image = cv2.filter2D(processed_image, -1, sharp_kernel)
            elif operation == "Canny Edge Detection":
                processed_image = cv2.Canny(processed_image, min_thresh, max_thresh)
            elif operation == "Threshold":
                _, processed_image = cv2.threshold(processed_image, thresh_value, 255, cv2.THRESH_BINARY)
            elif operation == "Rotate":
                (h, w) = processed_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed_image = cv2.warpAffine(processed_image, M, (w, h))
            elif operation == "Adjust Brightness/Contrast":
                processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast, beta=brightness)
            elif operation == "Color Adjustment (Hue/Saturation)":
                hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Update session state with processed image
            st.session_state.current_image = processed_image
        
        # Display the current image (original if None, processed otherwise)
        display_image = st.session_state.current_image
        channels = "BGR" if len(display_image.shape) == 3 else "GRAY"
        caption = "Original Image" if operation == "None" else f"Edited Image ({operation})"
        st.image(display_image, channels=channels, caption=caption, use_column_width=True)
        
        # Save option
        if st.button("Save Edited Image"):
            pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) if len(display_image.shape) == 3 else display_image)
            pil_image.save("edited_image.png")
            st.success("Image saved as 'edited_image.png'!")
        
        # Reset button
        if st.sidebar.button("Reset to Original"):
            st.session_state.current_image = image.copy()
            st.experimental_rerun()
    else:
        st.error("Error loading image. Please upload a valid image file.")
else:
    st.info("Please upload an image to get started.")
