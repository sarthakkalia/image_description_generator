import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from pickle import load
from PIL import Image
import pyttsx3
import joblib
import io

# Initialize the Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load the tokenizer and model
max_length = 32
tokenizer = load(open("tokenizer.p", "rb"))
model = joblib.load("model.sav")

# Load the Xception model for feature extraction
xception_model = Xception(include_top=False, pooling="avg")

# Global variable to store the last description
last_description = None


def extract_features(image_file, model):
    """
    Extract features from an image using the Xception model.
    """
    try:
        image = Image.open(image_file)
    except:
        raise ValueError("ERROR: Couldn't open image! Make sure the file is a valid image.")
    
    # Resize and preprocess the image
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[-1] == 4:  # Handle images with 4 channels (RGBA)
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0  # Normalize the image
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    """
    Map an integer to a word using the tokenizer's word index.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    """
    Generate a description for an image using the trained model.
    """
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def speak_text(command):
    """
    Speak the generated text description using text-to-speech.
    """
    engine = pyttsx3.init()
    try:
        engine.say(command)
        engine.runAndWait()
    except RuntimeError as e:
        print(f"Error: {e}")


def allowed_file(filename):
    """
    Check if the uploaded file is an allowed image type.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template("file.html")


@app.route('/file', methods=['GET', 'POST'])
def main_page():
    """
    Handle the image upload and prediction.
    """
    global last_description
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                # Read the image directly from the uploaded file
                img_bytes = io.BytesIO(file.read())
                photo = extract_features(img_bytes, xception_model)
                description = generate_desc(model, tokenizer, photo, max_length)
                last_description = description[5:-3]  # Remove 'start' and 'end' tokens
                return render_template('file.html', prediction=last_description)
            except Exception as e:
                return render_template('file.html', prediction=f"Error: {e}")
    return render_template('file.html', prediction=None)


@app.route('/audio')
def audio():
    """
    Speak the generated description using text-to-speech.
    """
    global last_description
    if last_description:
        try:
            speak_text(last_description)
            return render_template('file.html', prediction=last_description, audio_message="Audio played successfully.")
        except Exception as e:
            return render_template('file.html', prediction=last_description, audio_message=f"Error: {e}")
    else:
        return render_template('file.html', prediction=None, audio_message="No description available to play.")


if __name__ == '__main__':
    app.run(debug=True)
