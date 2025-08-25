import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

sz = 128

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "data2/train",
    target_size=(sz, sz),
    batch_size=10,
    color_mode="grayscale",
    class_mode="categorical",
)

test_set = test_datagen.flow_from_directory(
    "data2/test",
    target_size=(sz, sz),
    batch_size=10,
    color_mode="grayscale",
    class_mode="categorical",
)

classifier = tf.keras.models.Sequential()

classifier.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=[128, 128, 1],
    )
)


classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))


classifier.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
)

classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))


classifier.add(tf.keras.layers.Flatten())


classifier.add(tf.keras.layers.Dense(units=128, activation="relu"))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=96, activation="relu"))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=64, activation="relu"))
classifier.add(
    tf.keras.layers.Dense(units=27, activation="softmax")
)  # softmax for more than 2


classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)


classifier.summary()


classifier.fit(training_set, epochs=5, validation_data=test_set)


# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print("Model Saved")
classifier.save_weights("model-bw.weights.h5")
print("Weights saved")
