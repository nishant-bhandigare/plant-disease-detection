import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Loading the trained model
model = load_model('Plant Disease Prediction\plant_disease_detection.h5')

#Loading the prediction image
image_path=r"Plant Disease Prediction\Dataset\PlantVillage_Test\TomatoYellowCurlVirus4.JPG"

# Defining a function for disease prediction on a single image of a plant leaf
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    #eilf ladder for respective class indices
    if predicted_class==0:
        print("Plant Detected: Pepper Bell")
        print("Predicted Disease: Baterial Spot")
    elif predicted_class==1:
        print("Plant Detected: Pepper Bell")
        print("Predicted Disease: None, Plant is healthy")
    elif predicted_class==2:
        print("Plant Detected: Potato")
        print("Predicted Disease: Early Blight")
    elif predicted_class==3:
        print("Plant Detected: Potato")
        print("Predicted Disease: Late Blight")
    elif predicted_class==4:
        print("Plant Detected: Potato")
        print("Predicted Disease: None, Plant is healthy")
    elif predicted_class==5:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Bacterial Spot")
    elif predicted_class==6:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Early Blight")
    elif predicted_class==7:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Late Blight")
    elif predicted_class==8:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Leaf Mould")
    elif predicted_class==9:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Septoria Leaf Spot")
    elif predicted_class==10:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Spider Mites Two Spotted Spider Mite")
    elif predicted_class==11:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Target Spot")
    elif predicted_class==12:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Yellow Leaf Curl Virus")
    elif predicted_class==13:
        print("Plant Detected: Tomato")
        print("Predicted Disease: Mosaic Virus")
    elif predicted_class==14:
        print("Plant Detected: Tomato")
        print("Predicted Disease: None, Plant is healthy")

predict_disease(image_path)