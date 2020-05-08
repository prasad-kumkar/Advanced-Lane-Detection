import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.image as mpimg
import cv2

def transform(img, src, dst):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


if __name__ == "__main__":
    image = mpimg.imread("test_images/signs_vehicles_xygrad.png")
    src = np.float32([[500, 470],       #top left
                        [700, 470],     #top right
                        [900, 630],     #bottom right
                        [230, 630]])    #bottom left

    dst = np.float32([[250, 0],
                        [1000, 0],
                        [1000, image.shape[0]],
                        [250, image.shape[0]]])
    warped = transform(image, src, dst)

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax1.imshow(warped)
    ax2.imshow(image)
    plt.show()