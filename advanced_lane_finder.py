from hls_color_threshold import hls_select
import pickle
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
from sobel_abs import abs_sobel_thresh
from sobel_dir import dir_threshold
from sobel_mag import mag_thresh

from sliding_window import find_lane_pixels, fit_polynomial

image = cv2.imread('test_images/test3.jpg')

img_size = (image.shape[1], image.shape[0])
width = img_size[0]
height = img_size[1]

points = pickle.load( open("test_images/calibration_wide/wide_dist_pickle.p", "rb"))
mtx = points["mtx"]
dist = points["dist"]

#Distortion Correction
undist = cv2.undistort(image, mtx, dist, None, mtx)

#Color threshold
hls = hls_select(undist, thresh=(90, 255))

# Gradient threshold (NOT USED)
'''
ksize = 3 
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
'''
#Combining HLS and Sobel gradient (NOT USED)
'''combined2 = np.zeros_like(dir_binary)
combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls == 1)] = 1'''

# Using HLS thresholding as it is more cleaner

src = np.float32([[500, 470],       #top left
                    [700, 470],     #top right
                    [900, image.shape[0]],     #bottom right
                    [230, image.shape[0]]])    #bottom left

dst = np.float32([[250, 0],
                    [1000, 0],
                    [1000, image.shape[0]],
                    [250, image.shape[0]]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

warped = cv2.warpPerspective(hls, M, img_size, flags=cv2.INTER_LINEAR)

line = fit_polynomial(warped)
unwarp = cv2.warpPerspective(line, M_inv, img_size, dst=image, flags=cv2.INTER_LINEAR)


image = cv2.addWeighted(unwarp,0.4, undist, 1,0)

# plt.subplot(121)
# plt.imshow(unwarp)
# plt.subplot(122)

plt.imshow(image)
plt.show()