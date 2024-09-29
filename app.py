import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained face detection model from OpenCV's DNN module
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Color database for clothing recommendations
color_database = {
    "Very Fair": [
        {"name": "Light Blue", "hex": "#ADD8E6"},
        {"name": "Lavender", "hex": "#E6E6FA"},
        {"name": "Soft Pink", "hex": "#FFB6C1"},
        {"name": "Pastel Yellow", "hex": "#FFFFE0"},
    ],
    "Fair": [
        {"name": "Soft Peach", "hex": "#FFDAB9"},
        {"name": "Sea Green", "hex": "#2E8B57"},
        {"name": "Coral", "hex": "#FF7F50"},
        {"name": "Mint Green", "hex": "#98FF98"},
    ],
    "Medium": [
        {"name": "Olive Green", "hex": "#808000"},
        {"name": "Burgundy", "hex": "#800020"},
        {"name": "Burnt Orange", "hex": "#CC5500"},
        {"name": "Turquoise", "hex": "#40E0D0"},
    ],
    "Tan": [
        {"name": "Chocolate", "hex": "#D2691E"},
        {"name": "Khaki", "hex": "#F0E68C"},
        {"name": "Deep Blue", "hex": "#00008B"},
        {"name": "Rust", "hex": "#B7410E"},
    ],
    "Dark": [
        {"name": "Vibrant Red", "hex": "#FF0000"},
        {"name": "Dark Blue", "hex": "#00008B"},
        {"name": "Emerald Green", "hex": "#50C878"},
        {"name": "Purple", "hex": "#800080"},
    ],
    "Deep Dark": [
        {"name": "Jet Black", "hex": "#343434"},
        {"name": "Maroon", "hex": "#800000"},
        {"name": "Dark Magenta", "hex": "#8B008B"},
        {"name": "Dark Orange", "hex": "#FF8C00"},
    ]
}

# Function to detect faces
def detect_face(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            return face
    return None

# Improved skin tone detection based on specific facial regions
def detect_skin_tone(face_image):
    # Convert to HSV color space
    hsv_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin regions
    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    
    # Bitwise-AND mask and original image
    skin_pixels = cv2.bitwise_and(face_image, face_image, mask=skin_mask)

    # Convert masked skin pixels to RGB
    skin_pixels_rgb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2RGB)
    skin_pixels_rgb = skin_pixels_rgb[skin_mask != 0]  # Only keep skin pixels

    if skin_pixels_rgb.size == 0:  # Check if there are no skin pixels
        return "Unknown"  # Return a placeholder or handle as needed

    # Calculate the average color of skin pixels
    avg_color = np.mean(skin_pixels_rgb, axis=0)

    red, green, blue = avg_color

    # Improved skin tone classification based on RGB values
    if red > 220 and green > 190 and blue > 180:
        return "Very Fair"
    elif red > 200 and green > 170 and blue > 160:
        return "Fair"
    elif red > 150 and green > 120 and blue > 100:
        return "Medium"
    elif red > 120 and green > 90 and blue > 80:
        return "Tan"
    elif red > 80 and green > 60 and blue > 50:
        return "Dark"
    else:
        return "Deep Dark"

# Recommend clothing colors based on skin tone
def recommend_colors(skin_tone):
    return color_database.get(skin_tone, [])

# Display the recommended colors
def display_color_recommendations(colors):
    color_palette = [color["hex"] for color in colors]
    
    # Convert hex to RGB tuples
    rgb_palette = [(int(color[1:3], 16) / 255, int(color[3:5], 16) / 255, int(color[5:7], 16) / 255) for color in color_palette]

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow([rgb_palette])
    ax.set_axis_off()
    st.pyplot(fig)

# Streamlit interface
st.title("Skin Tone Detection and Clothing Color Recommendation")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Check if the image has an alpha channel (4 channels) and remove it
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect face and skin tone
    face = detect_face(image_np)
    if face is not None:
        skin_tone = detect_skin_tone(face)
        st.write(f"Detected Skin Tone: {skin_tone}")

        # Recommend clothing colors based on detected skin tone
        recommended_colors = recommend_colors(skin_tone)
        if recommended_colors:
            st.write("Recommended Clothing Colors:")
            display_color_recommendations(recommended_colors)
        else:
            st.write("No recommendations available for this skin tone.")
    else:
        st.write("No face detected in the image.")
