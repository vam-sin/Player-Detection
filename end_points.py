import cv2
import numpy as np
import time
from scipy import stats
from copy import deepcopy
from line_merge import lineMerge
from random import randint
from imageOperatives import *
from ground_color import *
# from lines import *


'''
  Req: This code requires sklearn to run.
  The code accepts image containing bw data of white lines extracted from parent image.
  Initially, the code scans the image sequentially using a window, and if the data in window seems likely to be a line,
  regression is applied on the window. If the reg score is high enough, the line segment is considered.
  Using this part of line, find_right_end_point and find_left_end_point are called which return the right and the left end points of the complete line.
  Once the end points are retrieved, the sequential search continues till the end of image is reached.
  coords contains two points on the line in the form [[y1,x1],[y2,x2]]
  '''

def find_right_end_point(image, sx, sy, edge, m):
    window = image[int(sx-0.5*edge): int(sx+0.5*edge), int(sy-0.5*edge): int(sy+0.5*edge)]  # to get the inital mask of line
    init_area = np.sum(window)
    mask = deepcopy(window)
    mask[mask!=0] = 1
    mask = cv2.dilate(mask, (np.ones((10,10), np.uint8)))     # mask is dilated, because the line we aim to find is generally slightly curved, if mask is not dilated we lose the line eventually
    break_encountered = 0       # storing if a break in the line is encountered.

    while True:
        if(int(sx-0.5*edge) < 0 or int(sx+0.5*edge) > 1080 or int(sy-0.5*edge) < 0 or int(sy+0.5*edge) > 1920):
            break

        window = np.multiply(image[int(sx-0.5*edge): int(sx+0.5*edge), int(sy-0.5*edge): int(sy+0.5*edge)], mask)
        area = np.sum(window)
        # print(area)
        # cv2.namedWindow('hmm', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('hmm', 1280,720)
        # cv2.rectangle(temp_img, (int(sy-0.5*edge), int(sx-0.5*edge)), (int(sy+0.5*edge), int(sx+0.5*edge)),255,1)
        # cv2.imshow('hmm', temp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if(area < init_area/7):
            if(break_encountered == 0):
                # y is the horizontal coordinate in the image and x is the vertical
                if(m<1):
                    sy += edge
                    sx += edge*m
                else:
                    sy += edge/m
                    sx += edge
                break_encountered = 1
                continue
            else:
                break
        break_encountered = 0

        last_box = window
        x_coord = int(sx - 0.5*edge)
        y_coord = int(sy - 0.5*edge)

        if(m<1):
            sy += edge
            sx += edge*m
        else:
            sy += edge/m
            sx += edge

    #finding end point of the line in the box.
    last_box = np.fliplr(last_box)
    nonzero = np.nonzero(last_box.T)
    return x_coord+nonzero[1][0], y_coord+edge-nonzero[0][0]


# similar to find_right_end_point
def find_left_end_point(image, sx, sy, edge, m):
    window = image[int(sx-0.5*edge): int(sx+0.5*edge), int(sy-0.5*edge): int(sy+0.5*edge)]
    init_area = np.sum(window)
    mask = deepcopy(window)
    mask[mask!=0] = 1
    mask = cv2.dilate(mask, (np.ones((10,10), np.uint8)))
    break_encountered = 0
    
    while True:
        if(int(sx-0.5*edge) < 0 or int(sx+0.5*edge) > 1080 or int(sy-0.5*edge) < 0 or int(sy+0.5*edge) > 1920):
            break
        window = np.multiply(image[int(sx-0.5*edge): int(sx+0.5*edge), int(sy-0.5*edge): int(sy+0.5*edge)], mask)
        area = np.sum(window)

        if(area < init_area/7):
            if(break_encountered == 0):
                if(m<1):
                    sy -= edge
                    sx -= edge*m
                else:
                    sy -= edge/m
                    sx -= edge
                break_encountered = 1
                continue
            else:
                break

        break_encountered = 0

        last_box1 = window
        x_coord1 = int(sx - 0.5*edge)
        y_coord1 = int(sy - 0.5*edge)

        if(m<1):
            sy -= edge
            sx -= edge*m
        else:
            sy -= edge/m
            sx -= edge

    nonzero1 = np.nonzero(last_box1.T)
    return x_coord1+nonzero1[1][0], y_coord1+nonzero1[0][0]


def getEndPoints(img):
    filter_size = 60
    lines = np.zeros(img.shape)
    # we use this as a canvas to print the complete lines we find from regression.
    # once a line has been found, we then do not consider any other window that containes that line, to reduce computation.

    x_min = np.array(np.where(img == 255))[:,0].T
    x_min = x_min[0]

    coords = []

    for i in range(int(img.shape[0]/filter_size)):
        if (i+1)*filter_size < x_min:
            continue
        for j in range(int(img.shape[1]/filter_size)):
            # break
            edge = filter_size
            window = img[i*edge:(i+1)*edge, j*edge:(j+1)*edge]

            area = np.sum(window)/255
            ratio = area/(filter_size**2)   # The area of white pixels upon bounding box area. A low value suggests that the box contains a line.
            # print(ratio)
            # break
            if(area > 60 and ratio < 0.18):
                if((np.sum(lines[i*edge:(i+1)*edge, j*edge:(j+1)*edge]) == 0)):     # This ensures that the same line isn't operated on twice
                    points = np.array(np.argwhere([window != 0])[::4])[:,1:3]

                    # [::4] is to take every fourth point only. Reducing number of points makes regression faster without any considerable accuracy loss.
                    x = points[:,0] + i*edge
                    y = points[:,1] + j*edge
                    slope, intercept, r_value, p_value, std_error = stats.linregress(y, x)
                    # print(r_value, std_error)
                    std_error = 1
                    if(abs(r_value) > 0.88 or (std_error > 0.5 and ((np.std(x)/np.std(y) > 5) or (np.std(x)/np.std(y) < 1/5)))):
                        if(abs(r_value) < 0.88):
                            if(np.std(x)/np.std(y) > 5):
                                slope = np.inf
                            else:
                                slope = 0
                        # print(slope)
                        # print(reg.coef_, reg.intercept_)
                        x1, y1 = find_right_end_point(img, (i+0.5)*edge, (j+0.5)*edge, edge, slope)
                        x2, y2 = find_left_end_point(img, (i+0.5)*edge, (j+0.5)*edge, edge, slope)
                        coords.append([[x1,y1], [x2,y2]])
                        cv2.line(lines, (y1,x1), (y2,x2),255,10)

    # plotting the coords with diff colors to get pairs of points.
    # print("COORDS : ",coords)
    return coords

###############

if __name__ == '__main__':
    time_init = time.time()
    img = cv2.imread('dst.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # rangeH,rangeS,rangeV = getGroundColor(img)
    # # print("["+str(rangeH)+"],["+str(rangeS)+"],["+str(rangeV)+"]")
    # # groundColorMask = rangeToMask(img,[rangeH],[rangeS],[rangeV])
    # groundColorMask = rangeToMask(img,[[0.26666666666666666, 0.35]],[[0.3686274509803922, 0.6627450980392157]],[[0.615686274509804, 0.8313725490196079]])
    # show(groundColorMask)

    # thres_val = getLineRange(img,groundColorMask)
    # print(thres_val)

    # lineMask = getLineMask(img,groundColorMask,thres_val)
    # show(lineMask)
    lineMask=img
    coords = getEndPoints(img)
    # print("coords= ", coords)
    coords,_ = lineMerge(coords)
    # coords=coords.astype(np.int_)
    # print("coords1= ", coords)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255, 100, 0), (113, 66, 244)]*2

    for p in range(len(coords)):
      point = coords[p]
      idx = randint(0, len(colors)-1)
      cv2.circle(lineMask,(point[0][1], point[0][0]), 10, colors[idx], -10)
      cv2.circle(lineMask,(point[1][1],point[1][0]), 10, colors[idx], -10)
      colors.pop(idx)

    print('Time = ', time.time() - time_init)


    cv2.imwrite('./points_final.png', lineMask)
