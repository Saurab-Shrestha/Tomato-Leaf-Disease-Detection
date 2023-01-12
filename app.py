import os
from flask import Flask, redirect, render_template, request
from PIL import Image
# import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
# Keras
# from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)

MODEL_PATH = "web app/potato.h5"

# load your trained model
model = load_model(MODEL_PATH)

class_names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def predict(model, img):
    img_array = image.load_img(img,target_size=(256,256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')



@app.route('/submit',methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('web app/static/uploads', filename)
        image.save(file_path)
        print(file_path)

        pred, confidence = predict(model, file_path)
        print(print)
        return render_template('submit.html',pred=pred,confidence=confidence)



if __name__ == '__main__':
    app.run(debug=True, port=5001)