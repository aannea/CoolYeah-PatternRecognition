import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob

def load_model(model_path='mymodel.h5'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_path, img_size):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_image(model, img_path, img_size=128):
    img_array = preprocess_image(img_path, img_size)
    prediction = model.predict(img_array)
    return prediction

if __name__ == '__main__':
    # Load the model
    model = load_model('mymodel.h5')

    # Path to the image to be predicted
    img_path = './test/no_damage/-95.17819_30.040034999999996.jpeg'  # Change this to the path of your image

    # Predict the image
    prediction = predict_image(model, img_path)

    # Print the prediction
    if prediction[0] > 0.5:
        print(f"Prediction: Damage ({prediction[0][0]})")
    else:
        print(f"Prediction: Not Damage ({prediction[0][0]})")
