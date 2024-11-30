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

# ----> The change: Set img_path directly
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=False, default='/content/drive/MyDrive/Colab Notebooks/Flickr8k_Dataset/Flicker8k_Dataset/111537222_07e56d5a30.jpg', help="Image Path")
# args = vars(ap.parse_args())
# img_path = args['image']
img_path = r'C:\Users\sai\PycharmProjects\image_discreption_generator\Flicker8k_Dataset\3637013_c675de7705.jpg' # Set the image path directly
# ----> End of the change

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

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")



import pyttsx3


print(description)
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()



SpeakText(description)
plt.imshow(img)
plt.show()