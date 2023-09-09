import os
import numpy as np
import cv2
from glob import glob

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from metrics import dice_coef, iou
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import pandas as pd
from tqdm import tqdm

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.tif")))
    masks = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    size = int(len(images) * split)

    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split, random_state=42)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    if x is None:
        raise ValueError(f"Failed to load mask at {path}")
    ori_x = x
    x = x / 255.0
    x = x>0.5
    x = x.astype(np.int32)
    return ori_x, x

def save_result(ori_x, ori_y, y_pred, save_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

    # Create the concatenated image
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)

    # Invert colors of the last image
    cat_images[:, -(W+10):, :] = 255 - cat_images[:, -(W+10):, :]

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path + '.jpg', cat_images)


if __name__=="__main__":
    create_dir("results_contour_model")

    """ Load Model """
    with CustomObjectScope({'iou':iou, 'dice_coef':dice_coef}):
        model=tf.keras.models.load_model("files_contours_01/model_contour_base_01.h5")

    """ Dataset """
    dataset_path = os.path.join("..", "..", "Augmented_contours", "dataset_augmented_01")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    """ Prediction and metrics values"""
    SCORE = []
    for i in tqdm(range(len(test_x)), desc="Processing images"):
        name = os.path.splitext(test_x[i])[0]

        """Reading the image and mask"""
        ori_x, x = read_image(test_x[i])
        ori_y, y = read_mask(test_y[i])

        """Prediction"""
        y_pred = model.predict(x)[0]>0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        save_path=f"./results_contours_02/{os.path.basename(name)}"
        save_result(ori_x, ori_y, y_pred, save_path)

        """ Flattening the numpy arrays """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average='binary')
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average='binary')
        recall_value = recall_score(y, y_pred, labels=[0, 1], average='binary')
        precision_value = precision_score(y, y_pred, labels=[0, 1], average='binary')

        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    score = pd.DataFrame(SCORE, columns=["Name", "Accuracy", "F1", "IoU", "Recall", "Precision"])
    score.to_csv("results_contour_model/scores_contour_model.csv")
