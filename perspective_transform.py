import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.image as mpimg
import cv2

image = mpimg.imread("test_images/signs_vehicles_xygrad.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height = gray.shape[0]
width = gray.shape[1]

src = np.float32([[500, 470],
                    [700, 470],
                    [900, 630],
                    [230, 630]])

dst = np.float32([[250, 0],
                    [1000, 0],
                    [1000, height],
                    [250, height]])

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)


plt.subplot(121)
plt.imshow(warped)
plt.subplot(122)
plt.imshow(image)
plt.show()