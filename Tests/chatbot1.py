import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow.keras as keras

# Langflow API Configuration
BASE_API_URL = "https://ethereal05engineer-plantdisease.hf.space"
FLOW_ID = "281516ad-f829-4eb1-b526-5b9a0472b928"

# Load Model - Cache each model individually
@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path, compile=False)

# CSS Styling
st.markdown("""
    <style>
    /* Sidebar styling */
    .css-18e3th9 {background-color: #f0f2f6; padding-top: 20px; border-radius: 10px;}
    /* Chat bubbles */
    .user-message {background-color: #e6f2ff; padding: 10px; border-radius: 10px; margin: 5px;}
    .assistant-message {background-color: #d6f5d6; padding: 10px; border-radius: 10px; margin: 5px;}
    /* Typewriter effect cursor */
    @keyframes blink { 50% { opacity: 0; } }
    .typewriter-cursor { border-right: 2px solid black; animation: blink 0.7s infinite;}
    </style>
""", unsafe_allow_html=True)

# Process chatbot response with Langflow API
def get_bot_response(user_input: str):
    api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
    payload = {"input_value": user_input, "output_type": "chat", "input_type": "chat"}
    response = requests.post(api_url, json=payload)
    return response.json() if response.status_code == 200 else {}

# Extract main response text
def process_response(bot_response_json):
    try:
        bot_message = bot_response_json["outputs"][0]["outputs"][0]["results"]["message"]["data"]
        bot_text = bot_message.get("text", "")
        table_data = pd.DataFrame(bot_message["table"]["data"], columns=bot_message["table"]["columns"]) if "table" in bot_message else None
        return bot_text, table_data
    except KeyError:
        return "I didn’t understand that. Could you rephrase?", None

# Typewriter effect
def typewriter_effect(bot_text):
    response_container = st.empty()
    full_text = ""
    for char in bot_text:
        full_text += char
        response_container.markdown(full_text + "<span class='typewriter-cursor'>|</span>", unsafe_allow_html=True)
        time.sleep(0.03)
    response_container.markdown(full_text, unsafe_allow_html=True)  # Final text without cursor

# Display bot response
def display_bot_response(bot_text, table_data):
    with st.chat_message("assistant"):
        typewriter_effect(bot_text)
        if table_data is not None:
            st.dataframe(table_data)

# Main disease detection functionality
def disease_detection_section(model, class_names):
    st.subheader("Plant Disease Detection")
    file_uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if file_uploaded:
        image = Image.open(file_uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((256, 256))
        test_image = np.expand_dims(np.array(image) / 255.0, axis=0)
        predictions = model.predict(test_image)
        confidence = round(100 * np.max(predictions[0]), 2)
        predicted_class = class_names[np.argmax(predictions)]

        st.progress(confidence / 100.0)  # Confidence level as progress bar
        st.write(f"Prediction: **{predicted_class}** ({confidence}%)")

# Chatbot Section
def chatbot_section():
    st.subheader("Chatbot for Plant Health Guidance")

    # Initialize chat state
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Toggle chat open state
    chat_button = st.button("Open Chatbot 💬")
    if chat_button:
        st.session_state.chat_open = not st.session_state.chat_open

    # Display chat interface if chat is open
    if st.session_state.chat_open:
        if st.button("Clear Chat"):
            st.session_state.messages = []

        # Display all chat messages
        for message in st.session_state.messages:
            style = "user-message" if message["role"] == "user" else "assistant-message"
            with st.chat_message(message["role"]):
                st.markdown(f"<div class='{style}'>{message['content']}</div>", unsafe_allow_html=True)

        # Chat input and bot response processing
        if prompt := st.chat_input("Ask me something..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

            bot_response_json = get_bot_response(prompt)
            bot_text, table_data = process_response(bot_response_json)

            if bot_text:
                display_bot_response(bot_text, table_data)
                st.session_state.messages.append({"role": "assistant", "content": bot_text})

# Sidebar for Plant Selection
st.sidebar.title("🌱 Select Plant Type")
st.sidebar.markdown("<small>Choose a plant type for disease detection:</small>", unsafe_allow_html=True)
plant_type = st.sidebar.radio("Choose a Plant", ["Potato 🥔", "Tomato 🍅", "Corn 🌽"])

# Display Disease Detection Section
if plant_type == "Potato 🥔":
    model = load_model("potato_model.h5")
    disease_detection_section(model, ["Potato:Early Blight", "Potato:Healthy", "Potato:Late Blight"])
elif plant_type == "Tomato 🍅":
    model = load_model("tomato_model.h5")
    disease_detection_section(model, ["Tomato Healthy","Tomato Mosaic Virus","Tomato Bacterial Spot","Tomato Eary Blight","Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot","Tomato Spider Mites","Tomato Target Spot","Tomato Yellow Leaf Curl Virus"])
elif plant_type == "Corn 🌽":
    model = load_model("corn_model.h5")
    disease_detection_section(model, ["Corn: Cercospora", "Corn: Common Rust","Corn: Healthy","Corn: Northern Leaf Blight"])

# Display Chatbot Section below the Disease Detection Section
st.divider()  # Add a divider for visual separation
chatbot_section()