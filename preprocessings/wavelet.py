import numpy as np
import cv2
import pywt

def wavelet(vv):
    # DWT at 2 scales (‘haar’ or ‘db2’)
    coeffs2 = pywt.wavedec2(vv.astype(np.float32), 'db2', level=2)
    # coeffs2 = [cA2, (cH2,cV2,cD2), (cH1,cV1,cD1)]
    cA2,(cH2,cV2,cD2),(cH1,cV1,cD1) = coeffs2
    # energy maps
    eng_low  = np.square(cA2)
    eng_high = (cH2**2 + cV2**2 + cD2**2)
    # resize to original
    eng_low_r  = cv2.resize(eng_low,  vv.shape[::-1])
    eng_high_r = cv2.resize(eng_high, vv.shape[::-1])
    # ratio and difference
    ratio = eng_low_r / (eng_high_r + 1e-6)
    diff  = eng_low_r - eng_high_r
    # normalize
    ch0 = cv2.normalize(vv,       None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    ch1 = cv2.normalize(ratio,    None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    ch2 = cv2.normalize(diff,     None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([ch0, ch1, ch2])