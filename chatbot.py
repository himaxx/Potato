import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import base64
import json
from typing import Optional
import warnings
from languages import LANGUAGES
import speech_recognition as sr
import pyttsx3
from streamlit.runtime.scriptrunner import add_script_run_ctx
from threading import Thread

# Langflow API Configuration
BASE_API_URL = "https://himmaannsshhuu-langflow.hf.space"
FLOW_ID = "8bfbf173-531c-430e-a670-b37d76822c91"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings

try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

# Tweaks dictionary to customize the flow
TWEAKS = {
    "ChatInput-BxwPy": {},
    "Prompt-4XAry": {},
    "OpenAIModel-6Yk3m": {},
    "ChatOutput-QHBDv": {}
}

def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run the flow with a given message and optional tweaks.
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}

    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Use the function
set_background('back.jpg')

@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path, compile=False)

# Function to convert text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech rate if needed
    engine.setProperty('volume', 0.9)  # Adjust volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

# Function to handle streaming bot response with text-to-speech
def display_bot_response_with_streaming(bot_text, table_data=None):
    response_container = st.empty()
    full_text = ""
    for char in bot_text:
        full_text += char
        response_container.markdown(full_text, unsafe_allow_html=True)
        time.sleep(0.03)  # Typewriter effect
    response_container.markdown(full_text, unsafe_allow_html=True)

    # Trigger voice synthesis in a separate thread for real-time playback
    thread = Thread(target=speak_text, args=(bot_text,))
    add_script_run_ctx(thread)
    thread.start()

    if table_data is not None:
        st.dataframe(table_data)

# Function to capture voice input
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
            return None

# Display the bot's response with a typewriter effect
def display_bot_response(bot_text, table_data):
    with st.chat_message("assistant"):
        display_bot_response_with_streaming(bot_text, table_data)  # Use the streaming function

# Improve image preprocessing and prediction
def disease_detection_section(model, class_names, lang):
    st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>CropSaviour: AI Based Plant Disease Detection and Assistance</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #FFFFFF;'>Identify Plant Diseases 🌿</h2>", unsafe_allow_html=True)
    
    # Use the drag-and-drop text
    file_uploaded = st.file_uploader(lang["upload_image"], type=["jpg", "jpeg", "png"], label_visibility="collapsed", help=lang["drag_and_drop"])
    
    if file_uploaded:
        try:
            image = Image.open(file_uploaded)
            st.image(image, caption="Uploaded Image", use_column_width=True, clamp=True)

            # Improve image preprocessing
            image = image.convert('RGB')  # Ensure RGB format
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            test_image = np.array(image) / 255.0
            test_image = np.expand_dims(test_image, axis=0)

            # Add error handling for prediction
            try:
                with st.spinner(lang["analyzing_image"]):
                    predictions = model.predict(test_image, verbose=0)
                confidence = round(100 * np.max(predictions[0]), 2)
                predicted_class = class_names[np.argmax(predictions)]

                st.success(f"**{lang['predicted_class']}:** {predicted_class} ({confidence}%)")
                st.progress(confidence / 100.0)
            except Exception as e:
                st.error(lang["error_prediction"].format(str(e)))
        except Exception as e:
            st.error(lang["error_processing"].format(str(e)))

# Make the chatbot section interactive
def chatbot_section(lang):
    # Initialize session state to store chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Voice input option
    if st.button("🎤 Speak to Chat"):
        user_input = voice_input()
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get bot response
            bot_response_json = run_flow(
                message=user_input,
                endpoint=ENDPOINT or FLOW_ID,
                tweaks=TWEAKS
            )
            bot_text, table_data = process_response(bot_response_json)

            if bot_text:
                # Display bot response
                display_bot_response(bot_text, table_data)
                # Add bot message to session state
                st.session_state.messages.append({"role": "assistant", "content": bot_text})

    # Chat input for text messages
    if prompt := st.chat_input(lang["chat_input"]):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        bot_response_json = run_flow(
            message=prompt,
            endpoint=ENDPOINT or FLOW_ID,
            tweaks=TWEAKS
        )
        bot_text, table_data = process_response(bot_response_json)

        if bot_text:
            # Display bot response
            display_bot_response(bot_text, table_data)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})

# Process the response received from the API
def process_response(bot_response_json):
    try:
        bot_message = bot_response_json["outputs"][0]["outputs"][0]["results"]["message"]["data"]
        bot_text = bot_message.get("text", "")
        table_data = pd.DataFrame(bot_message["table"]["data"], columns=bot_message["table"]["columns"]) if "table" in bot_message else None
        return bot_text, table_data
    except KeyError:
        return "I didn't understand that. Could you please repeat?", None

# Main disease detection and chatbot section
# Set default language
default_language = "en"
selected_language = st.sidebar.selectbox("Select Language", 
    ["English", "Hindi", "Marathi", "Bengali", "Malayalam", "Punjabi", "Gujarati"], 
    index=0)

# Map full language names to language codes
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Gujarati": "gu"
}

# Get the selected language code
selected_language_code = language_map[selected_language]

# Get the selected language dictionary
lang = LANGUAGES[selected_language_code]

st.sidebar.title(lang["title"])
plant_type = st.sidebar.radio(lang["title"], lang["plant_types"])

if plant_type == lang["plant_types"][0]:  # Potato
    model = load_model("model/potato_model.h5")
    disease_detection_section(model, ["Potato:Early Blight", "Potato:Healthy", "Potato:Late Blight"], lang)
elif plant_type == lang["plant_types"][1]:  # Tomato
    model = load_model("model/tomato_model.h5")
    disease_detection_section(model, ["Tomato_Healthy", "Tomato_mosaic_virus", "Tomato_bacterial_spot",
                                      "Tomato_early_blight", "Tomato_late_blight", "Tomato_leaf_mold",
                                      "Tomato_Septoria_leaf_spot", "Tomato_spider_mites", "Tomato_target_spot",
                                      "Tomato_yellow_leaf_curl_virus"], lang)
elif plant_type == lang["plant_types"][2]:  # Corn
    model = load_model("model/corn_model.h5")
    disease_detection_section(model, ["Corn: Cercospora", "Corn: Common Rust", "Corn Healthy", "Corn Northern Blight"], lang)

# Add button for chatbot section and toggle chat window
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# Custom CSS for button styling
st.markdown("""
<style>
.custom-button {
    background-color: #4CAF50; /* Green */
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    border-radius: 25px; /* Rounded corners */
    transition: background-color 0.3s, transform 0.3s; /* Smooth transition */
}

.custom-button:hover {
    background-color: #45a049; /* Darker green on hover */
    transform: scale(1.05); /* Slightly enlarge on hover */
}

.custom-button:active {
    transform: scale(0.95); /* Shrink on click */
}
</style>
""", unsafe_allow_html=True)

# Chat button
if st.button("Chat", key="chat_button"):
    st.session_state.chat_open = not st.session_state.chat_open

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []  # Clear the chat history
    st.success("Chat cleared!")  # Provide feedback to the user

# Save chat button
if st.button("Save Chat"):
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    with open("chat_history.txt", "w") as f:
        f.write(chat_history)
    st.success("Chat history saved!")  # Provide feedback to the user

if st.session_state.chat_open:
    chatbot_section(lang)