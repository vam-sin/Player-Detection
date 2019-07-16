import cv2
import numpy as np
from ground_color import *
from scipy import signal, stats
from copy import deepcopy
from DLdata import connectDB, disconnectDB, getFrameData, getPlayersMask

'''
If there is a split of brightness in the image due to sunlight, set sunlight_split to 1, else set it to 0.
Also, if sunlight_split = 1, lines are detected seperately from both parts and concatenated.
An intersection line often forms during the concatenation. It is removed if remove_intersection_line is 1.
In the very rare case that this removal degrades the final output, set remove_intersection_line to 0.
'''

def find_white_thres(image, thres_val, thres_adjust):
    # the function starts at thres_val, keeps decreasing it by thres_adjust, until any blobs start to emerge in the mask.
    while True:
        bw = np.zeros(image.shape)
        bw[ image > thres_val ] = 1
        kernel = np.ones((20,20),np.uint8)
        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        area = np.sum(opened)

        if area > 0:
            return thres_val+thres_adjust
            break

        thres_val -= thres_adjust


def adjust_white_thres(image, thres_val, thres_adjust):
    prev_green_boxes = 0
    prev_total_boxes = 0
    prev_box_ratio = 0
    thres_init = thres_val
    ratio_dict = {}
    while True:
        bw = np.zeros(image.shape)
        bw[ image >= thres_val ] = 255
        temp_img = deepcopy(bw)
        img = deepcopy(temp_img)
        temp_img = np.squeeze(np.stack((temp_img,) * 3, -1))

        filter_size = 60
        if(np.sum(img) != 0):
            x_min = np.array(np.where(img == 255))[:,0].T
            x_min = x_min[0]
        else:
            x_min = 1080

        total_boxes = 0
        green_boxes = 0
        box_ratio = 0

        for i in range(int(temp_img.shape[0]/filter_size)):
            if (i+1)*filter_size < x_min:
                continue
            for j in range(int(temp_img.shape[1]/filter_size)):
                # break
                edge = filter_size
                window = img[i*edge:(i+1)*edge, j*edge:(j+1)*edge]
                area = np.sum(window)/255
                ratio = area/(filter_size**2)   # The area of white pixels upon bounding box area. A low value suggests that the box contains a line.
                # print(ratio)
                # break
                if(area > 80):
                    cv2.rectangle(temp_img, (j*edge, i*edge), ((j+1)*edge, (i+1)*edge), (0,0,255),1)
                    total_boxes += 1
                    if(ratio < 0.18):
                        points = np.array(np.argwhere([window != 0])[::4])[:,1:3]
                        # [::4] is to take every fourth point only. Reducing number of points makes regression faster without any considerable accuracy loss.
                        x = points[:,0] + i*edge
                        y = points[:,1] + j*edge
                        slope, intercept, r_value, p_value, std_error = stats.linregress(y, x)
                        # print(r_value, std_error)
                        std_error = 1
                        stdx = np.std(x)
                        stdy = np.std(y)
                        if(abs(r_value) > 0.80 or (std_error > 0.5 and (stdy != 0 and ((stdx/stdy > 5) or (stdx/stdy < 1/5))))):
                            # cv2.rectangle(temp_img, (j*edge, i*edge), ((j+1)*edge, (i+1)*edge), (0,255,0),1)
                            green_boxes += 1

        prev_green_boxes = green_boxes if prev_green_boxes == 0 else prev_green_boxes
        prev_total_boxes = total_boxes if prev_total_boxes == 0 else prev_total_boxes

        prev_box_ratio = box_ratio if prev_box_ratio == 0 else prev_box_ratio

        box_ratio = green_boxes*(green_boxes/total_boxes) if total_boxes != 0 else 0

        # cv2.namedWindow('hmm', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('hmm', 1280,720)
        # cv2.imshow('hmm', temp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if((box_ratio < prev_box_ratio) or (green_boxes - prev_green_boxes) < -4):
        if((thres_val-thres_init) > 30):
            return (ratio_dict[max(ratio_dict.keys())])

        ratio_dict[box_ratio] = thres_val
        thres_val += thres_adjust
        prev_green_boxes = green_boxes
        prev_total_boxes = total_boxes
        prev_box_ratio = box_ratio


def find_v_thres(v_channel):
    histo = gethistogram(v_channel, 256)[1:]        # taking v_channel histogram excluding black

    # smoothing out the histogram by taking mean
    for i in range(2, len(histo)-2):
        histo[i] = (histo[i-2] + histo[i-1] + histo[i] + histo[i+1] + histo[i+2])/5

    peaks = signal.find_peaks(histo, prominence = 5e3, distance = 60)     # using the scipy function to find peaks in the histogram satisfying given parameters

    lower_peak = peaks[0][0]
    upper_peak = peaks[0][1]

    v_thres = lower_peak + np.argmin(histo[lower_peak : upper_peak])    # caluculating minimum valley between the two peaks

    return v_thres


def getStandMask(ground_mask, player_mask):
    kernel = np.ones((20,20),np.uint8)
    ground_mask = cv2.dilate(ground_mask, kernel)

    ground_mask = ground_mask.astype(np.uint8)
    num_components,labels,stats,centroids = cv2.connectedComponentsWithStats(ground_mask,connectivity = 8)
    prevCentroid = 0
    for i in range(num_components):
        area = stats[i][4]
        if area > 500000:
            if centroids[i][1] > prevCentroid:
                mask = i
                prevCentroid = centroids[i][1]
    stands_mask = np.zeros(ground_mask.shape)
    stands_mask[labels == mask] = 1
    stands_mask = stands_mask.astype(np.uint8)

    player_mask = 1-player_mask
    stands_mask[player_mask == 1] = 1

    kernel = np.ones((25,25),np.uint8)
    stands_mask = cv2.morphologyEx(stands_mask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((50,50),np.uint8)
    stands_mask = cv2.morphologyEx(stands_mask, cv2.MORPH_OPEN, kernel)

    return stands_mask


def getLineRange(img, ground_mask, player_mask, sunlight_split = 0, remove_intersection_line = 1):
    # img = cv2.imread("484.png", 1)
    # left, right, left2, right2, left3, right3 = getGroundColor(img)
    # ground_mask = rangeToMask(img,[[left/180,right/180]],[[left2/255,right2/255]],[[left3/255,right3/255]],0,0,0)
    # kernel = np.ones((15,15),np.uint8)
    # ground_mask = cv2.dilate(ground_mask, kernel)
    # kernel = np.ones((50,50),np.uint8)
    # player_mask = cv2.erode(ground_mask, kernel)
    # kernel = np.ones((80,80),np.uint8)
    # stands_mask = cv2.erode(cv2.dilate(player_mask, kernel), kernel)
    # kernel = np.ones((80,80),np.uint8)
    # stands_mask = cv2.dilate(cv2.erode(stands_mask, kernel), kernel)
    kernel = np.ones((20,20),np.uint8)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
    player_mask = np.multiply(ground_mask, player_mask)
    kernel = np.ones((80,80),np.uint8)
    stands_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((80,80),np.uint8)
    stands_mask = cv2.morphologyEx(stands_mask, cv2.MORPH_OPEN, kernel)

    bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw_img = np.multiply(player_mask, bw_img)
    # cv2.imwrite('gray.png', bw_img)

    thres_val_rough = find_white_thres(bw_img, 220, 10)         # to remove total iterations, first a rought value is taken
    thres_val = find_white_thres(bw_img, thres_val_rough, 1)    # minute iterations thereafter
    thres_val = adjust_white_thres(bw_img, thres_val, 1)

    bw_mask = np.zeros(bw_img.shape)
    bw_mask[ bw_img >= thres_val ] = 255
    bw_mask = bw_mask.astype(np.uint8)

    if sunlight_split == 0:
        # print("White range = ", [[thres_val, 255]])
        # cv2.imwrite('lines.png', bw_mask)
        lines = bw_mask
    else:
        # print("White range (upper half) = ", [[thres_val, 255]])
        part1 = bw_mask
        # part1 is the mask from the upper part of image

        v_channel = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)[:,:,2]
        v_channel = v_channel.astype(np.uint8)

        v_thres = find_v_thres(v_channel)

        v_mask = np.zeros(v_channel.shape)
        v_mask[v_channel < v_thres] = 1

        # closing v_mask, adding stands_mask, and closing again to get a mask for complete upper half including stands
        v_mask = cv2.dilate(v_mask, np.ones((50,50),np.uint8))
        v_mask = cv2.erode(v_mask, np.ones((100,100),np.uint8))
        v_mask = np.multiply(v_mask, stands_mask)
        v_mask = cv2.erode(v_mask, np.ones((150,150), np.uint8))
        v_mask = cv2.dilate(v_mask, np.ones((150,150), np.uint8))

        if remove_intersection_line == 1:
            kernel = np.ones((15,15),np.uint8)
            v_mask = cv2.erode(v_mask, kernel)

        # cv2.imwrite('v_mask.jpg', v_mask*255)

        section_mask = v_mask
        section = np.multiply(bw_img, section_mask)
        # cv2.imwrite('lower_section.png', section)

        thres_val_rough = find_white_thres(section, 220, 10)            # finding white thres for lower part
        thres_val = find_white_thres(section, thres_val_rough, 2)
        bw_mask = np.zeros(section.shape)
        bw_mask[ section >= thres_val ] = 255
        part2 = bw_mask

        # print("White range (lower half) = ", [[thres_val, 255]])

        lines = part1 + part2           # concatenating upper and lower part
        # cv2.imwrite('lines.png', lines)
    lines = lines.astype(np.uint8)

    return thres_val


def getLineMask(img, ground_mask, player_mask, threshold):
    # kernel = np.ones((15,15),np.uint8)
    # groundMask = cv2.dilate(groundColorMask, kernel)
    # kernel = np.ones((50,50),np.uint8)
    # player_mask = cv2.erode(groundMask, kernel)
    # kernel = np.ones((80,80),np.uint8)
    # stands_mask = cv2.erode(cv2.dilate(player_mask, kernel), kernel)
    # kernel = np.ones((80,80),np.uint8)
    # stands_mask = cv2.dilate(cv2.erode(stands_mask, kernel), kernel)
    kernel = np.ones((20,20),np.uint8)
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
    player_mask = np.multiply(ground_mask, player_mask)
    kernel = np.ones((80,80),np.uint8)
    stands_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((80,80),np.uint8)
    stands_mask = cv2.morphologyEx(stands_mask, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('player_mask.png', player_mask*255)
    # cv2.imwrite('stands_mask.png', stands_mask*255)
    # img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bw_img = np.multiply(player_mask, bw_img)

    bw_mask = np.zeros(bw_img.shape)


    bw_mask[ bw_img >= threshold ] = 255
    bw_mask = bw_mask.astype(np.uint8)

    return bw_mask

if __name__ == '__main__':
    import os
    os.chdir('..')
    filename = '362.jpg'
    img = cv2.imread(filename, 1)
    frameNum = filename.split('.')[0]
    dbDir = './DB/data10.db'
    db = connectDB(dbDir)
    playerData, ballData, goalData = getFrameData(db, frameNum)
    disconnectDB(db)
    playerMask = getPlayersMask(playerData, ballData)

    rangeH,rangeS,rangeV = getGroundColor(img)
    # print(rangeH,rangeS,rangeV)
    # left, right, left2, right2, left3, right3 = 53, 72, 23, 162, 87, 205
    groundColorMask = rangeToMask(img,[rangeH],[rangeS],[rangeV],0,0,0)
    # groundColorMask = rangeToMask(img,[[0.2611111111111111, 0.35]], [[0.37254901960784315, 0.7254901960784313]], [[0.4627450980392157, 0.7568627450980392]])
    # groundColorMask = rangeToMask(img,[[left/180, right/180]], [[left2/255, right2/255]], [[left3/255, right3/255]])
    # cv2.imwrite("gro.png",255*groundColorMask)
    # show(groundColorMask)
    # thres_val = 140
    thres_val = getLineRange(img,groundColorMask, playerMask)
    # print(thres_val)

    bw_mask = getLineMask(img, groundColorMask, playerMask, thres_val)
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', 1280,720)
    cv2.imshow('win',bw_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # kernel2 = np.ones((25,25),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    # cv2.imwrite("lin.png",bw_mask)
