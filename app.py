#Phase 1: Importing project dependencies
import os
import requests
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify

#Phase 2: Load pretrained model

#-------Loading model structure------
with open("fashion_model_flask.json", "r") as f:
    model_loaded = f.read()

model = tf.keras.models.model_from_json(model_loaded)

#-------Loading model weights------
model.load_weights("fashion_model_flask.h5")


#PHASE 3: Flash app
app = Flask(__name__)


@app.route("/api/v1/<string:img_name>", methods=["GET", "POST"])
def classify_image(img_name):
    upload_dir = "uploads/"
    image = tf.keras.preprocessing.image.load_img(upload_dir+img_name, target_size=(28,28),color_mode='grayscale')
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = np.array([image])
    image = image.reshape(1,784)


    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    prediction = model.predict(image)

    return jsonify({'Predicted as':classes[np.argmax(prediction[0])]})