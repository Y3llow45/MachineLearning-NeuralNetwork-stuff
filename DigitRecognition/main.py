import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1  
)
datagen.fit(x_train)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  
    tf.keras.layers.MaxPooling2D(2,2),  
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Prevents overfitting
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=3, validation_data=(x_test, y_test))

accuracy, loss = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}, Loss: {loss}')


for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense) and layer.activation.__name__ == 'softmax_v2':
        layer.activation = tf.keras.activations.softmax
model.save("./DigitRecognition/digit_recognition4.keras")

model = tf.keras.models.load_model("./DigitRecognition/digit_recognition4.keras")

for x in range(1, 11):
    img = cv.imread(f'./DigitRecognition/{x}.png', cv.IMREAD_GRAYSCALE)
    img = cv.bitwise_not(img)
    img = cv.resize(img, (28, 28))
    img = img / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)
    prediction = model.predict(img)
    print(f'Prediction: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
