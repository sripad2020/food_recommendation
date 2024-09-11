from flask import Flask, render_template, request, redirect, url_for
from keras.api.models import load_model
from keras.api.preprocessing import image
from keras.api.applications.inception_v3 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
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

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(351, 351))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_label(pred_index):
    return label_index_dict.get(pred_index, "Label not found")

def classify_image(img_path):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    decoded_prediction_index = np.argmax(predictions)
    prediction_label = get_label(decoded_prediction_index)
    return str(prediction_label)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            path=f"static/images/{file.filename}"
            print(path)
            img_path = os.path.join('static/images', file.filename)
            file.save(img_path)
            prediction = classify_image(img_path)
            return render_template('index.html', prediction=prediction, img_path=img_path)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
