#################road lines###

import matplotlib.pyplot as plt
import cv2
import numpy as np


def region_of_intrest(img,vertices):
    mask=np.zeros_like(img)
    channel_count=img.shape[2]
    match_mask_color=(255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image=cv2.bitwise_and(img, mask)
    return masked_image

img=cv2.imread("road.png")
image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(image.shape)
width = image.shape[1]
height = image.shape[0]
region_of_intrest1=[
    (0,height),
    (width/2, height/2),
    (width, height),
]

cropped_image=region_of_intrest(image,vertices=np.array([region_of_intrest1], np.int32))
    
plt.imshow(cropped_image)
plt.show()