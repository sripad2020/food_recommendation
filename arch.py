import streamlit as st
from keras.api.models import load_model
from keras.api.preprocessing import image
from keras.api.applications.inception_v3 import preprocess_input
import numpy as np
import os

# Load your trained model
model = load_model('food_classification_model.h5')

# Updated label dictionary
label_index_dict = {
    0: 'Baked Potato', 1: 'Crispy Chicken', 2: 'Donut', 3: 'Fries', 4: 'Hot Dog',
    5: 'Sandwich', 6: 'Taco', 7: 'Taquito', 8: 'apple_pie', 9: 'burger',
    10: 'butter_naan', 11: 'chai', 12: 'chapati', 13: 'cheesecake', 14: 'chicken_curry',
    15: 'chole_bhature', 16: 'dal_makhani', 17: 'dhokla', 18: 'fried_rice', 19: 'ice_cream',
    20: 'idli', 21: 'jalebi', 22: 'kaathi_rolls', 23: 'kadai_paneer', 24: 'kulfi',
    25: 'masala_dosa', 26: 'momos', 27: 'omelette', 28: 'paani_puri', 29: 'pakode',
    30: 'pav_bhaji', 31: 'pizza', 32: 'samosa', 33: 'sushi'
}

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(351, 351))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Get the label from the prediction index
def get_label(pred_index):
    return label_index_dict.get(pred_index, "Label not found")

# Classify the uploaded image
def classify_image(img_path):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    decoded_prediction_index = np.argmax(predictions)
    prediction_label = get_label(decoded_prediction_index)
    return str(prediction_label)

# Streamlit app
st.title("Food Classification App")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join("static", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(img_path, caption='Uploaded Image', use_column_width=True)
    prediction = classify_image(img_path)
    st.write(f"Prediction: {prediction}")