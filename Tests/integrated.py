import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import tensorflow_hub as hub
import time

# Multi-language Support
lang = st.sidebar.selectbox("Select Language", ["English", "Hindi"])
if lang == "English":
    st.title('Plant Disease Detection System')
else:
    st.title('पौधे की बीमारी पहचान प्रणाली')

# Hide Streamlit Menu and Footer
hide_streamlit_style = """
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Chatbot Bubble (not functional yet, only visual)
chatbot_style = """
    <style>
    .chatbot-bubble {
        position: fixed;
        bottom: 10px;
        right: 10px;
        width: 60px;
        height: 60px;
        background-color: #2ECC71;
        border-radius: 50%;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        color: white;
    }
    </style>
    <div class="chatbot-bubble">💬</div>
"""
st.markdown(chatbot_style, unsafe_allow_html=True)

# Navigation Bar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Potato", "Tomato", "Corn"])

# Dynamic Model Loading
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = keras.models.load_model(model_path, compile=False)
    return keras.Sequential([hub.KerasLayer(model, input_shape=(256, 256, 3))])

# CSS for image animation and resizing
image_style = """
    <style>
    .image-container img {
        width: 400px;
        height: auto;
        opacity: 0;
        animation: fadeIn 2s forwards;
        margin-bottom: 20px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
"""
st.markdown(image_style, unsafe_allow_html=True)

# Function to display image with animation and fixed size
def display_image(image):
    st.markdown(f'<div class="image-container"><img src="data:image/jpeg;base64,{image_to_base64(image)}" /></div>', unsafe_allow_html=True)

# Convert image to base64 format
import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = keras.preprocessing.image.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict plant disease
def predict_class(image, model, class_names):
    test_image = preprocess_image(image)
    predictions = model.predict(test_image)
    confidence = round(100 * np.max(predictions[0]), 2)
    prediction = class_names[np.argmax(predictions)]
    return prediction, confidence

# Disease Info for each plant
def get_disease_info(disease, plant_type):
    disease_info = {
        "Potato": {
            "Early_blight":{
                "title": "Potato Early Blight",
                "description": "Early Blight is a common fungal disease affecting potato plants. It primarily targets older leaves, causing small, dark brown to black spots with concentric rings that give a target spot appearance. These spots can cause the surrounding leaf tissue to yellow and eventually die. In severe cases, the disease can also affect stems and tubers, leading to dark, sunken lesions.",
                "recommendations":"Remove infected parts and apply fungicides.",
                "severity": "success"

            },
            "Healthy": {
                "title": "Healthy Potato Plant",
                "description": "Your potato plant appears healthy.",
                "recommendations": "Continue proper care.",
                "severity": "success"
            },
            "Late_blight": {
                "title": "Late Blight",
                "description": "Causes dark lesions on leaves and stems.",
                "recommendations": "Remove infected parts and apply fungicides.",
                "severity": "warning"
            }
        },
        "Tomato": {
            "Tomato_Healthy": {
                "title": "Healthy Tomato Plant",
                "description": "Your tomato plant appears healthy.",
                "recommendations": "Continue proper care.",
                "severity": "success"
            },
            "Tomato_mosaic_virus":{
                "title": "Tomato Mosaic Virus",
                "description": "Tomato Mosaic Virus causes mottled and distorted leaves, stunted growth, and sometimes yellow-green spots on fruits.",
                "recommendations": "Remove and destroy infected plants. Use virus-resistant varieties in future plantings. Control weeds and aphids that may spread the virus.",
                "severity": "error"
            },
            "Tomato_batcterial_spot":{
                "title": "Tomato Bacterial Spot",
                "description":"Bacterial spot causes small, dark, raised spots on leaves, stems, and fruits. Leaves may turn yellow and drop prematurely.",
                "recommendations": "Remove infected plant parts. Avoid overhead watering. Use copper-based fungicides as a preventive measure.",
                "severity": "warning"

            },
            "Tomato_early_blight":{
                "title": "Tomato Early Blight",
                "description":"Early blight causes dark brown spots with concentric rings on lower leaves, which may turn yellow and drop.",
                "recommendations":"Remove infected leaves. Improve air circulation. Apply fungicides preventively. Practice crop rotation.",
                "severity":"warning"
            },
            "Tomato_late_blight":{
                "title": "Tomato Late Blight",
                "description": "Late blight causes large, dark brown patches on leaves and stems, and can lead to rapid plant death.",
                "recommendations":"Remove and destroy infected plants immediately. Use fungicides preventively. Plant resistant varieties.",
                "severity":"error"

            },
            "Tomato_leaf_mold":{
                "title":"Tomato Leaf Mold",
                "description":"Leaf mold causes pale green to yellow spots on upper leaf surfaces and olive-green to gray fuzzy growth on lower surfaces.",
                "recommendations":"Improve air circulation. Reduce humidity. Remove infected leaves. Apply fungicides if severe.",
                "severity":"warning"
            },
            "Tomato_Septoria_leaf_spot":{
                "title": "Tomato Septoria Leaf Spot",
                "description": "Septoria leaf spot causes small, circular spots with dark borders and light centers on lower leaves.",
                "recommendations": "Remove infected leaves. Improve air circulation. Apply fungicides preventively. Practice crop rotation.",
                "severity":"warning"
            },
            "Tomato_Spider_Mites": {
                "title": "Spider Mites",
                "description": "Causes stippling on leaves, turns yellow or bronze.",
                "recommendations": "Use insecticidal soaps or neem oil.",
                "severity": "warning"
            },
            "Tomato_target_spot":{
                "title":"Target Spot",
                "description": "Target spot causes brown, circular lesions with concentric rings on leaves, stems, and fruits.",
                "recommendations":"Remove infected plant parts. Improve air circulation. Apply fungicides preventively.",
                "severity":"warning"
            },
            "Tomato_yellow_leaf_curl_virus":{
                "title":"Yellow Leaf Curl Virus",
                "description":"Yellow leaf curl virus causes leaves to become small, curled, and yellow. Plants may be stunted with reduced fruit set.",
                "recommendations":"Remove and destroy infected plants. Control whiteflies (virus vectors). Use virus-resistant varieties.",
                "severity":"error" 
            }
        },
        "Corn": {
            
            "Cercospora": {
                "title": "Cercospora Leaf Spot",
                "description": "Causes pale brown or gray lesions on leaves.",
                "recommendations": "Use resistant hybrids and fungicides.",
                "severity": "warning"
            },
            "Common_rust":{
                "title": "Corn Common Rust",
                "description": "Form Rust Like Patches",
                "recommendations":"Use resistent ",
                "severity": "warning"

            },
            "Healthy": {
                "title": "Healthy Corn Plant",
                "description": "Your corn plant appears healthy.",
                "recommendations": "Continue proper care.",
                "severity": "success"
            },
            "Northern_leaf_blight":{
                "title": "Corn Northern Blight",
                "description": "Northern leaf blight causes long, elliptical, grayish-green to tan lesions on the leaves. As the disease progresses, these lesions may become darker and produce spores, giving them a dusty appearance",
                "recommendations": "- Use resistant hybrids.\n- Implement crop rotation with non-host crops.\n- Apply fungicides if the disease is severe and detected early.\n- Remove and destroy crop debris after harvest.\n- Improve air circulation in the field.",
                "severity": "error"
            }
        }
    }
    
    return disease_info[plant_type].get(disease, {})

# Main section for each plant
def main_section(plant_type, model_path, class_names):
    model = load_model(model_path)
    
    file_uploaded = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    if file_uploaded:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image:")
        display_image(image)
        
        with st.spinner('Analyzing image...'):
            time.sleep(2)  # Simulate prediction time
            result, confidence = predict_class(image, model, class_names)
        
        st.success(f'Prediction: {result} (Confidence: {confidence}%)')

        # Display Disease Information
        disease_info = get_disease_info(result, plant_type)
        if disease_info:
            st.markdown(f"### {disease_info['title']}")
            st.markdown(f"*Description:* {disease_info['description']}")
            st.markdown(f"*Recommendations:* {disease_info['recommendations']}")
        else:
            st.markdown("No information available for this disease.")

# Section Logic
if section == "Potato":
    st.header("Potato Disease Detection")
    main_section("Potato", "final_model.h5", ["Early_Blight","Healthy", "Late_blight"])
elif section == "Tomato":
    st.header("Tomato Disease Detection")
    main_section("Tomato", "tomato_model.h5", ["Tomato_Healthy", "Tomato_mosaic_virus", "Tomato_bacterial_spot","Tomato_early_blight", "Tomato_late_blight", "Tomato_leaf_mold","Tomato_Septoria_leaf_spot", "Tomato_spider_mites", "Tomato_target_spot","Tomato_yellow_leaf_curl_virus"])
elif section == "Corn":
    st.header("Corn Disease Detection")
    main_section("Corn", "corn1.h5",["Cercospora", "Common_rust",  "Northern_leaf_blight","Healthy"] )
