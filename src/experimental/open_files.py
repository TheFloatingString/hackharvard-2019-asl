import glob, os
from pathlib import Path
import numpy as np
from PIL import Image

temp_list_of_img = []
temp_list_of_labels = []

for file in Path("data").glob("**/*.jpg"):
	temp_list_of_img.append(str(file))
	temp_list_of_labels.append(str(file).split("\\")[-1][0])

list_of_img = [temp_list_of_img[x*100] for x in range(870)]
list_of_labels = [temp_list_of_labels[x*100] for x in range(870)]

def load_image(filename):
	img = Image.open(filename)
	img.load()
	image_array = np.asarray(img, dtype="int32")
	return np.array(image_array)

img_arrays = [np.true_divide(load_image(x),255) for x in list_of_img]
np.save("data/processed_images/processed_images.npy", img_arrays)

NUMBER_OF_CLASSES = len(np.unique(list_of_labels))
TYPES_OF_CLASSES = list(np.unique(list_of_labels))

list_of_label_encodings = list()

for label in list_of_labels:
	temp_list = [0]*NUMBER_OF_CLASSES
	location = TYPES_OF_CLASSES.index(label)
	temp_list[location] = 1
	list_of_label_encodings.append(np.array(temp_list))

np.save("data/training_labels/labels.npy", np.array(list_of_label_encodings))


# np.save("data/training_labels/labels.npy", list_of_labels)