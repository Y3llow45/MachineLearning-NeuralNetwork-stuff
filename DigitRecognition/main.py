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
  rotation_range=15,  
  width_shift_range=0.15,  
  height_shift_range=0.15,  
  zoom_range=0.1,
  shear_range=0.10,
)
datagen.fit(x_train)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),  
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate=0.0005,
  decay_steps=10000,
  decay_rate=0.95
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./DigitRecognition/best_model.keras', save_best_only=True)

history = model.fit(
  datagen.flow(x_train, y_train, batch_size=64), 
  epochs=20, validation_data=(x_test, y_test), 
  callbacks=[early_stopping, model_checkpoint])

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}, Loss: {loss}')

model = tf.keras.models.load_model("./DigitRecognition/best_model.keras")

for x in range(1, 11):
    img = cv.imread(f'./DigitRecognition/{x}.png', cv.IMREAD_GRAYSCALE)
    if img is None:
      print(f"Failed to load image {x}.png")
      continue
    img = cv.bitwise_not(img)
    img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
    img = img / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)
    prediction = model.predict(img, verbose=0)
    print(f'Prediction: {np.argmax(prediction)}')
    confidence = np.max(prediction) * 100
    print(f'Confidence: {confidence:.2f}%')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
