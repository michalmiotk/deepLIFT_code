import cv2
import numpy as np
image = cv2.imread("elephant.jpeg")
mod_image = np.empty_like(image)
height,width, channel = image.shape
for h in range(height):
    for w in range(width):
        for c in range(channel):
            new_pixel = image[h,w,c] + np.random.randint(-8,9)
            if 0<=new_pixel<=255:
                mod_image[h,w,c] = new_pixel
            else:
                mod_image[h,w,c] = image[h,w,c]

cv2.imwrite("elephant_mod.jpeg", mod_image)