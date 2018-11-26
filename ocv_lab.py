import cv2 as cv
import numpy as np
import copy


def Clamp(value, min, max):
    if value < min:
        return int(min)
    if value > max:
        return int(max)
    return int(value)


# Show algorithm steps
steps_show = False

# 1. Loading source image
src = cv.imread('Lenna.png')

height = src.shape[0]
width = src.shape[1]

if (steps_show):
    cv.imshow('src image', src)
    cv.waitKey(1000)

#########################
# 2. Transform to grayscale
#########################
gray_dst = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
if (steps_show):
    cv.imshow('grayscale image', gray_dst)
    cv.waitKey(1000)

#########################
# 3.Improve contrast
#########################
contrast_dst = cv.equalizeHist(gray_dst)
if (steps_show):
    cv.imshow('Improve contrast image', contrast_dst)
    cv.waitKey(1000)

#########################
# 4.gaussian blur for canny
#########################
# gauss = cv.GaussianBlur(copy.deepcopy(contrast_dst), (5, 5), 0)
# Canny edge detection
canny = cv.Canny(contrast_dst, 30, 200)
if (steps_show):
    cv.imshow('Canny edge detection image', canny)
    cv.waitKey(1000)

#########################
# 5. Corners detection
########################
corners = cv.goodFeaturesToTrack(copy.deepcopy(contrast_dst), 200, 0.01, 10)
corners_dst = copy.deepcopy(canny)
# draw circles in corners
for corner in corners:
    x, y = corner.ravel()
    cv.circle(corners_dst, (x, y), 3, 255, -1)

if (steps_show):
    cv.imshow('Corners detection image', corners_dst)
    cv.waitKey(1000)

#########################
# 6. Distance transform
#########################

corners_tmp = copy.deepcopy(corners_dst)

corners_tmp = cv.threshold(corners_tmp, 100, 255, cv.THRESH_BINARY)[1]
cv.bitwise_not(corners_tmp, corners_tmp)  # make only edges picture


dist_transform = cv.distanceTransform(corners_tmp, cv.DIST_L2, 3)

# dist_transform = cv.normalize(dist_transform, dist_transform, 0, 1., cv.NORM_MINMAX)

steps_show = True
if (steps_show):
    cv.imshow('Distance transform image', cv.normalize(copy.deepcopy(dist_transform),
                                                       copy.deepcopy(dist_transform)
                                                       , 0, 1., cv.NORM_MINMAX))
    cv.waitKey(1000)

####################(
# 7-8.Filter with integral image
####################


k = 0.8
integralImage = cv.integral(gray_dst)

filtImage = np.zeros((height, width), np.uint8)
print("start_filtering")

for y in range(0, height):
    for x in range(0, width):
        radius = min(int(k * dist_transform[y, x]), 4)

        A = integralImage[Clamp(y - radius, 0, height - 1), Clamp(x - radius, 0, width - 1)]
        B = integralImage[Clamp(y + 1 + radius, 0, height - 1), Clamp(x - radius, 0, width - 1)]
        C = integralImage[Clamp(y - radius, 0, height - 1), Clamp(x + 1 + radius, 0, width - 1)]
        D = integralImage[Clamp(y + 1 + radius, 0, height - 1), Clamp(x + 1 + radius, 0, width - 1)]

        sum = A + D - C - B
        ave = (2 * radius + 1) ** 2
        filtImage[y, x] = (sum / ave)

print("finish_filtering")

cv.imshow('Contrast image', contrast_dst)
cv.waitKey(1000)

cv.imshow('Filter image', filtImage)
cv.waitKey(0)
cv.destroyAllWindows()
