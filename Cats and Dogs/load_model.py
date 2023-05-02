from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np

model = load_model('Cats and Dogs\my_model.h5')

img_path = 'Cats and Dogs/test.png'
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_tensor = tf.keras.utils.img_to_array(img)
img_tensor /= 255.

# Add batch dimension
img_tensor = np.expand_dims(img_tensor, axis=0)

# Predict the class of the image
prediction = model.predict(img_tensor)

if prediction[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")
