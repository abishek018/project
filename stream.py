import streamlit as st
import cv2
import numpy as np

def apply_log_transformation(image):
    # Applying log transformation
    log_transformed = np.log1p(image.astype(np.float32))
    return (255 * (log_transformed / np.max(log_transformed))).astype(np.uint8)

def apply_dog_processing(image):
    # Apply some processing related to "dog" (you can replace this with your specific processing)
    # For demonstration, let's just blur the image
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_canny_edge_detection(image):
    # Applying Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def rotate_image(image, angle):
    # Rotating the image
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def mirror_image(image):
    # Mirroring the image
    return cv2.flip(image, 1)

def main():
    st.title("Image Processing with Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Image processing options
        processing_option = st.selectbox("Select Image Processing Option", ["Log Transformation", "Dog Processing", "Canny Edge Detection", "Rotate", "Mirror"])

        # Additional parameters for rotation
        rotation_angle = st.slider("Rotation Angle (degrees)", -180, 180, 0)

        # Apply the selected processing
        st.header(processing_option)
        if processing_option == "Log Transformation":
            processed_image = apply_log_transformation(image)
        elif processing_option == "Dog Processing":
            processed_image = apply_dog_processing(image)
        elif processing_option == "Canny Edge Detection":
            processed_image = apply_canny_edge_detection(image)
        elif processing_option == "Rotate":
            processed_image = rotate_image(image, rotation_angle)
        elif processing_option == "Mirror":
            processed_image = mirror_image(image)

        st.image(processed_image, caption=f"{processing_option} Result", use_column_width=True)

if __name__ == "__main__":
    main()
