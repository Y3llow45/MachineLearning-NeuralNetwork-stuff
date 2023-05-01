import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image

# Load cat and dog images
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/PetImages',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compile
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# train
model.fit(train_generator, epochs=10)

# evaluate
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '/path/to/test/directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator)

print('Test accuracy:', test_acc * 100, '%')

# predict on your own image
img_path = 'test.png'
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor /= 255.

# Add batch dimension
img_tensor = np.expand_dims(img_tensor, axis=0)

# Predict the class of the image
prediction = model.predict(img_tensor)

if prediction[0][0] > 0.5:
    print("Dog")
else:
    print("Cat")
