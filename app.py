import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import base64
from io import BytesIO

# -----------------------------
# 1. Page Configuration & Styles
# -----------------------------
st.set_page_config(page_title="Liver Fibrosis Classification", layout="centered")

# Custom CSS for styling and scanning animation
STYLES = """
<style>
body {
    background-color: #F8F8F8;
    color: #333333;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}
.title {
    font-size: 48px !important;
    color: #4B72FA;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
}
.subtitle {
    font-size: 18px !important;
    color: #666666;
    text-align: center;
    margin-bottom: 40px;
}
.footer {
    position: fixed;
    left: 0; 
    bottom: 0;
    width: 100%;
    text-align: center;
    color: #999999;
    font-size: 14px;
    padding: 10px 0;
    background-color: #FAFAFA;
    border-top: 1px solid #EEE;
}
.prediction-box {
    margin-top: 20px;
    text-align: center;
    font-size: 28px; /* Bigger font size */
    font-weight: bold;
    color: #333;
}
.prediction-class {
    color: #f1c40f; /* Bright yellow color for the predicted class */
    font-weight: 700;
}
/* Container for the scanning overlay */
.scan-container {
    position: relative;
    width: 100%;
    max-width: 500px; /* Adjust as needed */
    margin: 0 auto;
}
.scanned-image {
    width: 100%;
    height: auto;
}
.scan-overlay {
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg,
                                rgba(255,255,255,0) 0%,
                                rgba(255,255,255,0.6) 50%,
                                rgba(255,255,255,0) 100%);
    animation: scanning 2s infinite;
}
@keyframes scanning {
    0% {
        left: -100%;
    }
    100% {
        left: 100%;
    }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# -----------------------------
# 2. Title & Description
# -----------------------------
st.markdown("<h1 class='title'>Liver Fibrosis Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to predict its liver fibrosis class.</p>", unsafe_allow_html=True)

# -----------------------------
# 3. Model Preparation
# -----------------------------
@st.cache_resource  # Cache the loaded model to avoid repeated loads
def load_model(model_path: str):
    """Create and load the DenseNet121 model with trained weights."""
    try:
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 5)  # 5 classes
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Adjust this path as needed
MODEL_PATH = r"C:\Users\sajal\Desktop\work\SEPM Project\Dataset aug\liver_disease_densenet121_optimized.pth"
model = load_model(MODEL_PATH)

# Class names (ensure these match the training dataset order)
target_names = [
    "Cirrhosis",
    "No Fibrosis",
    "Periportal Fibrosis",
    "Portal Fibrosis",
    "Septal Fibrosis"
]

# -----------------------------
# 4. Image Transformations
# -----------------------------
image_size = 160
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_scanning_overlay_html(pil_image):
    """
    Convert a PIL image to a base64-encoded string and embed
    it in HTML with a scanning overlay.
    """
    # Convert PIL to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Return HTML code with scanning overlay
    html_code = f"""
    <div class="scan-container">
      <img src="data:image/png;base64,{img_str}" class="scanned-image" />
      <div class="scan-overlay"></div>
    </div>
    """
    return html_code

# -----------------------------
# 5. Upload & Display
# -----------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert file to a PIL image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display the uploaded image (static, no animation yet)
        static_image_placeholder = st.empty()
        static_image_placeholder.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Button to classify
        if st.button("Classify"):
            # 1) Replace the static image with scanning overlay
            scanning_html = generate_scanning_overlay_html(image)
            static_image_placeholder.markdown(scanning_html, unsafe_allow_html=True)
            
            # 2) Wait for scanning effect (2 seconds)
            time.sleep(2)
            
            # 3) Remove scanning overlay & show the final image again
            static_image_placeholder.image(image, caption="Scanned Image", use_container_width=True)
            
            # 4) Perform classification
            img_tensor = val_transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
            
            predicted_class = target_names[pred.item()]
            
            # 5) Display the prediction in big, yellow text
            st.markdown(
                f"<div class='prediction-box'>Predicted Class: "
                f"<span class='prediction-class'>{predicted_class}</span></div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# -----------------------------
# 6. Footer
# -----------------------------
st.markdown(
    "<div class='footer'>© 2025 Liver Fibrosis Classification App</div>", 
    unsafe_allow_html=True
)
