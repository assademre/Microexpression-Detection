import os
import cv2
import pickle
import random
import numpy as np

ABSPATH: str = os.path.abspath("Micro_Expressions")
SUBFOLDERS: list = ["test", "train"]
CATEGORIES = os.listdir(ABSPATH + "/train")
IMG_SIZE: int = 60  # new size of each image
MODEL_SAVE_NAME: str = f"micro_expressions"


def create_the_model() -> None:
    """
    The function takes the images from given folder and turns each of the image into arrays with the associated category
    """
    for subfolder in SUBFOLDERS:
        for category in CATEGORIES:
            first_path = os.path.join(ABSPATH, subfolder)
            path = os.path.join(first_path, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                # appending the original image
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                model_data.append([new_array, class_num])
                # appending the rotated image
                height, width = new_array.shape[:2]
                center = (width / 2, height / 2)
                rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
                rotated_image = cv2.warpAffine(src=new_array, M=rotate_matrix, dsize=(width, height))
                model_data.append([rotated_image, class_num])
    random.shuffle(model_data)


if __name__ == "__main__":
    model_data = list()
    X = list()
    y = list()
    create_the_model()

    for features, label in model_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    y = np.array(y)

    pickle_out = open(f"X_{MODEL_SAVE_NAME}.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(f"y_{MODEL_SAVE_NAME}.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
