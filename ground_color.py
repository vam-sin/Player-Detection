import cv2
import numpy as np
from sklearn.cluster import MeanShift,estimate_bandwidth
import pymeanshift as pms
from copy import deepcopy
import time
from lines import *

def rangeToMask(img,rangeH,rangeS,rangeV,toleranceH=0, toleranceS=0, toleranceV=0):
    HSV=rgb_to_hsv(img)
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H=HSV[:,:,0]
    S=HSV[:,:,1]
    # V=HSV[:,:,2]/255

    mask = np.zeros(img[:,:,0].shape)
    maskH = np.zeros(img[:,:,0].shape)
    maskS = np.zeros(img[:,:,0].shape)
    # maskV = np.zeros(img[:,:,0].shape)
    for i in range(0,len(rangeH)):
        maskH [ ((H > (rangeH[i][0]-toleranceH)) & (H <= (rangeH[i][1]+toleranceH))) ] = 1

    for i in range(0,len(rangeS)):
        maskS [ ((S > (rangeS[i][0]-toleranceS)) & (S <= (rangeS[i][1]+toleranceS))) ] = 1

    # for i in range(0,len(rangeV)):
    #     maskV [ ((V > (rangeV[i][0]-toleranceV)) & (V <= (rangeV[i][1]+toleranceV))) ] = 1

    # mask [ (maskH>=1) & (maskS>=1) & (maskH>=1) ] = 255
    mask [ (maskH>=1) & (maskS>=1) ] = 1
    # kernel = np.ones((7,7),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = mask.astype(np.uint8)
    return mask

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if max_val - min_val == 0:
        out = im.astype('float')/255;
    else:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def rgb_to_hsv(rgb):
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # print(np.max(r))
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

    v = maxc

    deltac = maxc - minc
    maxc[maxc == 0] = 1      #!!! to remove warning
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack([h, s, v])
    return res.reshape(input_shape)


def getlr(counts, thres):
    maxEl=np.amax(counts[1:])
    index=np.where(counts ==maxEl)

    left_flag=0
    right_flag=0
    starti=1
    right=index[0][0]+starti-1
    left=index[0][0]+starti-1
    index=index[0][0]+starti-1

    for i in range(0,np.size(counts)):
        if (index-i>0 and counts[index-i]>thres and left_flag != 1):
            left= index-i
        else:
            left_flag=1
        if (index+i<np.size(counts) and counts[index+i]>thres and right_flag != 1):
            right=index+i
        else:
            right_flag=1
        if (left_flag==1 and right_flag==1):
            break

    return left, right

def gethistogram(arr, maxval):
    unique, counts = np.unique(arr, return_counts=True)
    new=np.zeros(maxval)
    for i in range(len(unique)):
        new[unique[i]] = counts[i]

    return np.array(new)


def print_histogram( items ):
    i = 0
    for n in items:
        output = ''
        times = n
        # print(i, end='')
        i+=1
        while( times > 0 ):
          output += '*'
          times = times - 1
        # print(output)


def getGroundColor(img):
    img=np.array(img)
    pq=deepcopy(img)
    ic2=np.ones((1080,1920))
    I=im2double(img)
    I = np.array(I)
    I=I.astype(np.float32)

    print("Calculating ground color range...")

    HSV=rgb_to_hsv(I)
    H=HSV[:,:,0]
    H=np.reshape(H,(1080*1920,1))
    S=HSV[:,:,1]
    S=np.reshape(S,(1080*1920,1))
    V=HSV[:,:,2]
    V=np.reshape(V,(1080*1920,1))

    hsv=np.concatenate((H,S),axis=1)

    hsv=hsv.astype(np.float32)
    one=np.ones((1080,1920))

    time_init = time.time()

    init_centers = np.array([[np.median(H), np.median(S)], [0, 0]])

    labels = np.random.randint(2, size=(hsv.shape[0], 1), dtype=np.int32)
    labels = labels.astype(np.int32)
    for i in range(1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
        flags = cv2.KMEANS_RANDOM_CENTERS

        sumd,idx,center = cv2.kmeans(data = hsv,K = 2, bestLabels = labels,criteria = criteria, attempts = 1, flags=cv2.KMEANS_USE_INITIAL_LABELS, centers = init_centers)
        # sumd,idx,center = cv2.kmeans(hsv,2,None,criteria,1,flags)
        median1=np.median(H)
        # print(idx)
        ic=np.reshape(idx,(img.shape[0],img.shape[1]))

        ic=ic+one
        if(abs(center[0][0]-median1)<=abs(center[1][0]-median1)):
            ic[np.where(ic==1)]=1
            ic[np.where(ic==2)]=0
        else:
            ic[np.where(ic==1)]=0
            ic[np.where(ic==2)]=1

        ic=ic.astype(np.uint8)
        # ret2,ic = cv2.threshold(ic,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #
        # nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(ic,connectivity = 8)
        # sizes = stats[1:,-1]
        # filt = np.zeros((output.shape))
        # for i in range(0,nb_components-1):
        #     if(sizes[i] >= 500 and sizes[i] <= 5000000):
        #         filt[ output == i+1 ] = 255
        # filt = filt.astype(np.uint8)
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
        # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
        #
        # ic = cv2.morphologyEx(filt,cv2.MORPH_OPEN,kernel1)
        # ic = cv2.morphologyEx(ic, cv2.MORPH_CLOSE, kernel2)

        # ic=ic/255
        # print(ic)
        ic2=np.multiply(ic,ic2)
        #
        # cv2.imshow('abc', ic*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print('Time = ', time.time()-time_init)

    # cv2.imwrite("stands_mask.png",ic2*255)

    ic2 = np.squeeze(np.stack((ic2,) * 3, -1))
    new_im = np.multiply(pq,ic2)

    image = np.array(new_im)
    image = image.astype(np.uint8)

    (segmented_image, labels_image, number_regions) = pms.segment(image, spatial_radius=6, range_radius=4.5, min_density=50)


    lb = labels_image
    # cv2.imwrite("segmented_image.png",segmented_image)
    center = []

    time_init = time.time()

    final1=np.zeros((1080,1920))

    for i in range(0, number_regions):
        points = np.where(labels_image == i)
        # print(points)
        center = segmented_image[points[0][0]][points[1][0]][0]
        # print(center)
        area = len(points[0])

        if (center < 0.01):
            pass
        else:
            if (area > 10000):
                final1[lb==i] = 1

    print('Time = ', time.time()-time_init)

    final1=final1.astype(np.uint8)
    final3 = np.squeeze(np.stack((final1,) * 3, -1))
    # print(np.dtype(final3))

    final_im=np.multiply(new_im,final3)
    final_im=final_im.astype(np.uint8)
    # cv2.imwrite("ground_segment.png", final_im)

    HSV = cv2.cvtColor(final_im, cv2.COLOR_RGB2HSV)

    H=HSV[:,:,0]
    H=np.reshape(H,(1080*1920,1))

    S=HSV[:,:,1]
    S=np.reshape(S,(1080*1920,1))

    V=HSV[:,:,2]
    V=np.reshape(V,(1080*1920,1))


    thres=500

    counts1=gethistogram(H, 180)
    left, right = getlr(counts1, thres)

    ################

    counts2=gethistogram(S, 256)
    left2, right2 = getlr(counts2, thres)

    ################

    counts3=gethistogram(V, 256)
    left3, right3 = getlr(counts3, thres)

    ################

    mask = rangeToMask(pq,[[left/180,right/180]],[[left2/255,right2/255]],[[left3/255,right3/255]],0,0,0)
    pq=pq.astype(np.uint8)
    mask=mask.astype(np.uint8)
    mask = np.squeeze(np.stack((mask,) * 3, -1))
    print("Ground color range:")
    print([[left,right]])
    print([[left2,right2]])
    print([[left3,right3]])
    mask_im=np.multiply(pq,mask)

    cv2.imwrite("output.png",mask_im)

    return [left/180,right/180],[left2/255,right2/255],[left3/255,right3/255]


if __name__ == '__main__':

    rangeH=[0.2777777777777778, 0.39444444444444443] 
    rangeS=[0.35294117647058826, 0.7294117647058823] 
    rangeV=[0.41568627450980394, 0.7490196078431373]
    img = cv2.imread("./../Videos/imgs1/45.jpg", 1)
    groundColorMask = rangeToMask(img,[rangeH],[rangeS],[rangeV])
    # cv2.imwrite("ground_mask_test.jpg",groundColorMask*255)

    stands_mask=getStandMask(groundColorMask)
    # cv2.imwrite("stands_mask_test.jpg",stands_mask*255)
    # getGroundColor(img)
