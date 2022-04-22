import cv2
import numpy as np
import tensorflow as tf
from model_creating import CATEGORIES, IMG_SIZE


def prepare(filename: str) -> np.ndarray:
    """
    The function resize and turn to grayscale the given image
    :filename: the image to be resized and be turned grayscale
    """
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("32x2-model.model")
prediction = model.predict([prepare('test_image.png')])
numpy_list = prediction.astype(int)[0]
index, = np.where(numpy_list == 1)
print(CATEGORIES[int(index[0])])
