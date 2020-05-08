import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("test_images/wide_dist_pickle.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y


def corners_unwarp(img, nx, ny, mtx, dist):
       
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret:
        img = cv2.drawChessboardCorners(undist, (nx,ny), corners, ret)
        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
                '''Note: you could pick any four of the detected corners 
                 as long as those four corners define a rectangle
                 One especially smart way to do this would be to use four well-chosen
                 corners that were automatically detected during the undistortion steps
                 We recommend using the automatic detection of corners in your code '''
        offset = 100
        dst = np.float32([[offset, offset], 
                        [gray.shape[1]-offset, offset], 
                        [gray.shape[1]-offset, gray.shape[0]-offset],
                        [offset, gray.shape[0]-offset]
        ])
        M = cv2.getPerspectiveTransform(src, dst)                               #to get M, the transform matrix
        warped = cv2.warpPerspective(undist, M, (gray.shape[1],gray.shape[0]))  # to warp your image to a top-down view
        return warped, M
    return 0, 0

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()