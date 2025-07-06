import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/game_model.h5'
IMG_SIZE = 224  # Assuming EfficientNetB0 size

# Create app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model(MODEL_PATH)

# Dummy class labels (replace with your actual class list from LabelEncoder)
class_names = ['American Football','Basketball','Soccer', 'Tennis','Volleyball' ]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process and predict
            img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            return render_template('index.html', image=file.filename, result=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
