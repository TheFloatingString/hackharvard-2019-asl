import glob, os
from pathlib import Path
import numpy as np
from PIL import Image

temp_list_of_img = []
temp_list_of_labels = []

for file in Path("data").glob("**/*.jpg"):
	temp_list_of_img.append(str(file))
	temp_list_of_labels.append(str(file).split("\\")[-1][0])

list_of_img = [temp_list_of_img[x*1000] for x in range(80)]
list_of_labels = [temp_list_of_labels[x*1000] for x in range(80)]

def load_image(filename):
	img = Image.open(filename)
	img.load()
	image_array = np.asarray(img, dtype="int32")
	return image_array

img_arrays = [load_image(x) for x in list_of_img]
np.save("data/processed_images/processed_images.npy", img_arrays)
np.save("data/training_labels/labels.npy", list_of_labels)