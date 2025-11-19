import numpy as np
import cv2
import matplotlib.pyplot as plt


NORM_MEAN = (0.33791857209157045, 0.19346805519799193, 0.622593908173095)
NORM_STD = (0.15856856990599727, 0.11384739698653763, 0.29835953086232453)
NODATA_RGB = [255, 255, 0]

def convert(src):
    valid = src[src > 0]
    src[src <= 0] = np.max(src)
    src = (src - np.mean(valid)) / np.std(valid)

    C = np.sign(src) * np.log(np.abs(src) + 1)

    f = cv2.filter2D(src, -1, np.ones((5, 5)) / 25)
    s = src**2 - 2 * src * f - f**2

    E = cv2.erode(C, np.ones((5, 5)), iterations=2)

    R = C
    G = C / (abs(E) + 1)
    B = s

    R = (((np.clip(R, -2, 1.25) - (-0.8954522013664246)) / 0.515359103679657) - (-2.1432583332061768)) / (4.163023948669434 - (-2.1432583332061768))
    G = (((np.clip(G, -0.75, 1) - (-0.3864753842353821)) / 0.19613634049892426) - (-1.7169846296310425)) / (7.068936347961426 - (-1.7169846296310425))
    B = (((np.clip(B, -5, 0) - (-1.8781756162643433)) / 1.4927695989608765) - (-2.091296672821045)) / (1.2581818103790283 - (-2.091296672821045))

    rgb = np.empty((src.shape[0], src.shape[1], 3))
    rgb[:, :, 0] = np.clip(R * 255, 0, 255).astype("uint8")
    rgb[:, :, 1] = np.clip(G * 255, 0, 255).astype("uint8")
    rgb[:, :, 2] = np.clip(B * 255, 0, 255).astype("uint8")

    return rgb