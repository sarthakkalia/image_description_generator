import tensorflow as tf
from flask import Flask, render_template, request, url_for,send_from_directory,Response
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import joblib
import tensorflow as tf

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import argparse #No need for argparse in this context
from tensorflow.keras.models import load_model
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception




app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}





def extract_features(filename, model):
    try:
        image = Image.open(filename)

    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))



# Load the model with the custom layer
model=joblib.load("model.sav")

xception_model = Xception(include_top=False, pooling="avg")





import pyttsx3


def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']






@app.route('/')
def home():
    return render_template("file.html")



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            img_path="uploads\\"+filename
            photo = extract_features(img_path, xception_model)
            img = Image.open(img_path)
            global description
            description = generate_desc(model, tokenizer, photo, max_length)
            return render_template('file.html', prediction=description[5:-3],file_url=file_url)

    return render_template('file.html', prediction=None,file_url=None)


@app.route('/audio2')
def audio():
    SpeakText(description[5:-3])
    return render_template('file.html', prediction=description[5:-3],file_url=None)



if __name__ == '__main__':
    app.run(debug=True)

