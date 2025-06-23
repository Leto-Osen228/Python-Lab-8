import cv2
import numpy as np
import os

image = cv2.imread("Python-Lab-8/images/variant-5.jpg")
mean = 3
sigma = 25
noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
noised_image = cv2.add(image, noise)
cv2.imshow("Noised Image", noised_image)
cv2.waitKey(0)