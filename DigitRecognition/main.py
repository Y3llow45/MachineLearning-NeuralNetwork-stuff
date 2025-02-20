import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

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

model.fit(x_train, y_train, epochs=3)

accuracy, loss = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}, Loss: {loss}')


for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense) and layer.activation.__name__ == 'softmax_v2':
        layer.activation = tf.keras.activations.softmax
model.save("./DigitRecognition/digit_recognition3.keras")

model = tf.keras.models.load_model("./DigitRecognition/digit_recognition3.keras")

for x in range(1, 11):
    img = cv.imread(f'./DigitRecognition/{x}.png', cv.IMREAD_GRAYSCALE)
    img = cv.bitwise_not(img)
    img = cv.resize(img, (28, 28))
    img = tf.keras.utils.normalize(img, axis=1)
    img = np.array(img).reshape(-1, 28, 28)
    prediction = model.predict(img)
    print(f'Prediction: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
