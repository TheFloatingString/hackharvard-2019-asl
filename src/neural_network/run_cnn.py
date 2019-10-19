from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

import numpy as np

processed_images = np.load("data/processed_images/processed_images.npy")

INPUT_SHAPE = processed_images.shape

print(processed_images.shape)
print(processed_images[1])

labels = np.load("data/training_labels/labels.npy")
print(labels)

model = Sequential()
model.add(Conv2D(16, (12,12), input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=(8,8)))
model.add(Conv2D(16, (12,12)))
model.add(MaxPooling2D(pool_size=(8,8)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(29, activation="softmax"))

model.compile(loss="categorical_crossentropy",
				optimizer="sgd",
				metrics=["accuracy"])

model.fit(processed_images, labels, epochs=2, batch_size=64, validation_split=0.2)

model.save_weights("saved_model.h5")
