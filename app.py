
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename #secure the file
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save() in step1
MODEL_PATH = 'vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function() # Inbuilt function , to be used for imagenet models for prediction


#Function doing the uploaded image preprocessing
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image , converting image to array
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)#Expanding the dimension

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)#imported from imagenet_utils

    preds = model.predict(x)
    return preds

#app route for get method , here i give my root folder .Inside get method im uploading index.html
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')#'render_template' imported from flask utils, This shows your starting page

#app route for prediction.
@app.route('/predict', methods=['GET', 'POST'])#id="btn-predict" redirects to here
def upload():#upload image function
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']#based on 'name ' in upload code in index.html: <input type="file" name="file" id="imageUpload" accept=".png, .jpg, .jpeg">

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)#basepath gives root path/url of the working location
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))#joining makes this saves in uploads folder with filename
        f.save(file_path)#saving in filepath

        # Make prediction
        preds = model_predict(file_path, model)# model predict function defined on top . here i get class index

        # Process your result for human understanding
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode # decode_predictions library imported from imagenet_utils
                                                        #decode_predictions map class index to class labels
        result = str(pred_class[0][0][1])               # Convert to string, this locatio will have name of class
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)