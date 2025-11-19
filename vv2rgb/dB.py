import numpy as np
import cv2
import matplotlib.pyplot as plt


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
NODATA_RGB = [255, 255, 255]

def convert(vv):
    vv[vv <= 0] = np.max(vv)
    db = np.multiply(np.log10(vv), 20)

    db_norm = cv2.normalize(db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(db_norm, cv2.COLOR_GRAY2RGB)