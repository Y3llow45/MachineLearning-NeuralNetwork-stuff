import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'Cats and Dogs\PetImages',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'Cats and Dogs\PetImages',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(224, 224, 3)
    ))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='my_project'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'Cats and Dogs\PetImagesTest',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

tuner.search_space_summary()

tuner.search(train_generator, validation_data=validation_generator, epochs=10)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = build_model(best_hps)

model.fit(train_generator, validation_data=validation_generator, epochs=10)

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc * 100, '%')

model.save('better_model')
