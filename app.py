import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time

# Hide Streamlit components
hide_streamlit_style = """
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set page title
st.title('Potato Leaf Disease Prediction')

# Load the model once to avoid reloading on every image upload
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'final_model.h5'
    model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

model = load_model()

def display_image(image):
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    st.pyplot(figure)

def preprocess_image(image):
    image = image.resize((256, 256))
    image = keras.preprocessing.image.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image):
    test_image = preprocess_image(image)
    predictions = model.predict(test_image)
    class_names = ['Potato__Early_blight', 'Potato__healthy', 'Potato__Late_blight']
    confidence = round(100 * np.max(predictions[0]), 2)
    prediction = class_names[np.argmax(predictions)]
    return prediction, confidence

def get_disease_info(disease):
    if disease == 'Potato__Early_blight':
        return st.warning(
            "### Early Blight (Alternaria solani)\n\n"
            "**Description:**\n"
            "Early Blight is a common fungal disease affecting potato plants. It primarily targets older leaves, "
            "causing small, dark brown to black spots with concentric rings that give a 'target spot' appearance. "
            "These spots can cause the surrounding leaf tissue to yellow and eventually die. In severe cases, the disease "
            "can also affect stems and tubers, leading to dark, sunken lesions.\n\n"
            "**Management:**\n"
            "- **Crop Rotation:** Avoid planting potatoes or related crops in the same soil consecutively.\n"
            "- **Resistant Varieties:** Choose potato varieties that are resistant to Early Blight.\n"
            "- **Proper Spacing:** Ensure adequate spacing between plants to promote good air circulation.\n"
            "- **Sanitation:** Remove and destroy infected plant debris to reduce the source of fungal spores.\n"
            "- **Fungicides:** Apply fungicides preventively, especially during warm and humid conditions. "
            "Follow the recommended application rates and intervals.", icon= "⚠️"
        )
    elif disease == 'Potato__Late_blight':
        return st.error(
            "### Late Blight (Phytophthora infestans)\n\n"
            "**Description:**\n"
            "Late Blight is a severe fungal disease that can devastate potato crops. It causes water-soaked, pale green to dark brown lesions "
            "on leaves, which can expand rapidly. Under moist conditions, a white, fluffy mold may develop on the undersides of leaves. "
            "The disease can also spread to stems and tubers, causing dark, granular rot inside the tubers.\n\n"
            "**Management:**\n"
            "- **Resistant Varieties:** Use potato varieties that are resistant to Late Blight.\n"
            "- **Proper Spacing:** Plant potatoes with enough space to ensure good air circulation.\n"
            "- **Irrigation Management:** Avoid overhead irrigation and water early in the day to allow plants to dry quickly.\n"
            "- **Sanitation:** Remove and destroy infected plant material. Avoid leaving cull piles near growing areas.\n"
            "- **Fungicides:** Apply fungicides regularly, especially during cool, wet weather. Adhere to recommended application rates and schedules.", icon= "🚨"
        )
    else:
        return st.success(
            "### Healthy Plant\n\n"
            "**Description:**\n"
            "A healthy potato plant has vibrant green leaves without any spots, lesions, or discoloration. The foliage is uniform and free of wilting or mold. "
            "Healthy plants have robust growth and produce high-quality tubers without any signs of disease or rot.\n\n"
            "**Maintenance Tips:**\n"
            "- **Regular Monitoring:** Regularly inspect plants for any signs of disease or pest infestation.\n"
            "- **Proper Watering:** Water plants adequately, avoiding waterlogged conditions which can promote fungal diseases.\n"
            "- **Balanced Fertilization:** Use appropriate fertilizers to ensure the plants receive necessary nutrients.\n"
            "- **Good Hygiene:** Maintain garden hygiene by removing weeds and plant debris that can harbor pests and diseases.\n"
            "- **Crop Rotation:** Practice crop rotation to reduce the buildup of soil-borne diseases and pests.", icon = "✅"
        )

def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        display_image(image)
        
        with st.spinner('Making prediction...'):
            time.sleep(2)  # Simulate time taken for prediction
            result, confidence = predict_class(image)
        
        st.success(f'Prediction: {result} with {confidence}% confidence')

        disease_info = get_disease_info(result)
        st.markdown(disease_info, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
