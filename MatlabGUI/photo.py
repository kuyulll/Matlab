from ultralytics import YOLO
import numpy as np
import cv2
import os


def img_pre(img_path,model_path):
    model=YOLO(model_path)
    results=model(img_path)
    names_dict=results[0].names
    probs=results[0].probs.data.tolist()
    return names_dict[np.argmax(probs)]
