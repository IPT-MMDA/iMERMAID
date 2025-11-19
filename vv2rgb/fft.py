import numpy as np
import cv2

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (1, 1, 1)
NODATA_RGB = [0, 0, 0]

def get_early_preproc(src):
    img = src.copy()

    valid = np.isfinite(img) * (img > 0) * (img < 0.2)
    valid = cv2.morphologyEx(valid.astype("uint8"), cv2.MORPH_OPEN, np.ones((5, 5)), iterations=2).astype("bool")

    invalid = ~valid
    
    mean = np.mean(img[valid])
    std = np.std(img[valid])

    img[invalid] = mean
    # img should be in range [0, 1] for correct fft
    img = 0.5 * (1 + np.clip((img - mean) / std, -3, 3) / 3)

    return img, invalid

def to_fft(src):
    fft_image = np.fft.fft2(src)
    return np.fft.fftshift(fft_image) 

def from_fft(src):
    ifft_shifted = np.fft.ifftshift(src)  # Shift back the zero frequency to the original position
    filtered_image = np.fft.ifft2(ifft_shifted)
    return np.abs(filtered_image)

def fft_noise_removal(image, threshold_ratio=0.3):
    # Step 1: Forward FFT to transform the image to the frequency domain
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift zero frequency to the center

    # Step 2: Compute the power spectrum
    power_spectrum = np.abs(fft_shifted) ** 2

    # Step 3: Determine a threshold to filter out noise
    # Sort power values and determine threshold based on the ratio
    power_values = np.sort(power_spectrum.ravel())  # Flatten and sort the power spectrum
    threshold_index = int(threshold_ratio * len(power_values))  # Determine the index for the threshold
    threshold = power_values[threshold_index]  # Get the power value at the threshold

    # Step 4: Create a mask based on the threshold
    mask = power_spectrum > threshold  # Keep frequencies above the threshold

    # Apply the mask to the FFT-transformed image
    filtered_fft = fft_shifted * mask

    # Step 5: Perform the inverse FFT to transform the image back to the spatial domain
    ifft_shifted = np.fft.ifftshift(filtered_fft)  # Shift back the zero frequency to the original position
    filtered_image = np.fft.ifft2(ifft_shifted)
    filtered_image = np.abs(filtered_image)  # Get the magnitude (real part)

    return filtered_image

def create_hipass_strange(src, cutoff_frequency):
    rows, cols = src.shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol] = 0
    return mask

def convert(src):
    initial, invalid = get_early_preproc(src)
    noise_removed = fft_noise_removal(1 - initial)
    fft = to_fft(noise_removed)
    step1 = fft * create_hipass_strange(fft, 320)
    imgs = from_fft(step1)

    r = 1 - initial
    g = noise_removed
    b = noise_removed - imgs

    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    r[invalid] = 0
    g[invalid] = 0
    b[invalid] = 0

    return cv2.merge([r, g, b])