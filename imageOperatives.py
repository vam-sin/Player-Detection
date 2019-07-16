import cv2
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)

'''
Utility function to show an image
'''

def show(*imgs):
    fig=plt.figure(figsize=(20, 20))
    columns = math.ceil(np.sqrt(len(imgs)))
    rows = math.ceil(len(imgs)/columns)
    i = 1
    for img in imgs:
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
        i+=1
    plt.show()

'''
Utility function to show an image as plot
'''

def showPlot(img, linePts):
    for pt in linePts:
        print(pt)
        # plt.scatter(pt[0][0],pt[0][1])
        # plt.scatter(pt[1][0],pt[1][1])
        # plt.plot([pt[0][0],pt[1][0]],[pt[0][1],pt[1][1]])
        plt.scatter(pt[0][1],pt[0][0])
        plt.scatter(pt[1][1],pt[1][0])
        plt.plot([pt[0][1],pt[1][1]],[pt[0][0],pt[1][0]])
    plt.imshow(img)
    plt.show()

def rgb_to_hsv(rgb):
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
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

'''
Utility function to find L2 norm of vector
'''

def find_norm(vect):
    sumofsq=0
    for i in vect:
        sumofsq=sumofsq+i**2
    return np.sqrt(sumofsq)

'''
Utility function to convert image pixels to range [0,1]
'''

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if max_val - min_val == 0:
        out = im.astype('float')/255
    else:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def rangeToMask(img,rangeH,rangeS,rangeV,toleranceH=0,toleranceS=0,toleranceV=0):
    HSV=rgb_to_hsv(img)
    H=HSV[:,:,0]
    S=HSV[:,:,1]
    V=HSV[:,:,2]
    mask = np.zeros(img[:,:,0].shape)
    maskH = np.zeros(img[:,:,0].shape)
    maskS = np.zeros(img[:,:,0].shape)
    maskV = np.zeros(img[:,:,0].shape)

    for i in range(0,len(rangeH)):
        maskH [ ((H > (rangeH[i][0]-toleranceH)) & (H < (rangeH[i][1]+toleranceH))) ] = 255

    for i in range(0,len(rangeS)):
        maskS [ ((S > (rangeS[i][0]-toleranceS)) & (S < (rangeS[i][1]+toleranceS))) ] = 255

    for i in range(0,len(rangeV)):
        maskV [ ((V > (rangeV[i][0]-toleranceV)) & (V < (rangeV[i][1]+toleranceV))) ] = 255

    # mask [ (maskH>=1) & (maskS>=1) & (maskH>=1) ] = 255
    mask [ (maskH>=1) & (maskS>=1) ] = 1
    mask = mask.astype(np.uint8)
    return mask

def zero_norm(v):
    v=v>0
    v=np.array(v)
    L0=np.sum(v)

def getGroundObjects(groundColorMask):

    #Thresholding
    # mask = cv2.cvtColor(groundColorMask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur((255*groundColorMask),(5,5),0)
    ret2,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Breaking into Connected Components
    nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(mask,connectivity = 8)
    sizes = stats[1:,-1]

    #Components with area in range (10000 and 5000000) is taken only such that complete audience is removed
    mask = np.zeros((output.shape))
    for i in range(0,nb_components-1):
        if(sizes[i] >= 10000 and sizes[i] <= 5000000):
            mask[ output == i+1 ] = 255
    mask = mask.astype(np.uint8)
    ## mask now has ground - players

    #removing players from ground
    kernel = np.ones((20,20),np.uint8)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((70,70),np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = mask2.astype(np.uint8)
    ## mask2 now has just the ground

    mask = np.invert(mask)
    mask = np.multiply(mask,mask2)
    kernel = np.ones((7,7),np.uint8)
    # cv2.imwrite("./wclo.jpg",255*mask)
    # show(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = mask.astype(np.uint8)
    ## mask contains only players and lines on ground
    return mask

def linesAtAngle(groundLinesMask, slope, rotation, deviation):
    # print("IN LINESATANGLE\n")
    # print(slope,rotation,deviation, " <<<<<")
    rotation = abs(rotation)
    lines = cv2.HoughLinesP(groundLinesMask ,rho = 1,theta = 1*np.pi/180,threshold = 400,minLineLength = 400,maxLineGap = 10)
    # show(groundLinesMask)
    uniqueLines = mergelines(lines)
    # print(uniqueLines)
    ang1 = np.arctan(slope)*180/np.pi

    outputLines = []
    for line in uniqueLines:
        x1,y1,x2,y2 = line
        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        if(x2 != x1):
            slope2 = (y2-y1)/(x2-x1)
            ang2 = np.arctan(slope2)*180/np.pi
        else:
            ang2 = 90
        angle = abs(ang2-ang1)
        if(angle > 90):
            angle = 180 - angle
        if(angle > rotation - deviation and angle < rotation + deviation):
            length = lengthOf(line)
            line = np.append(line,length)
            outputLines.append(line)

    outputLines = np.array(outputLines)
    print(outputLines)
    print("EXIT\n")
    return outputLines

##used in linesAtAngle for merging multiple lines at same place
def lengthOf(line):
    x1,y1,x2,y2 = line
    length = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    return length

def mergelines(lines):
    uniqueLines = []
    flag = True
    if lines is None:
        return []
    for line in lines:
        for x1,y1,x2,y2 in line:
            for l in uniqueLines:
                if((abs(l[0]-x1)<=20 and abs(l[1]-y1)<=20) or (abs(l[2]-x2)<=20 and abs(l[3]-y2)<=20)):
                    if(lengthOf(l) < lengthOf(line[0].tolist())):
                        uniqueLines.remove(l)
                        uniqueLines.append(line[0].tolist())
                    flag = False
                    break
                else:
                    flag = True
            if(flag):
                uniqueLines.append(line[0].tolist())
    return uniqueLines

def extractComponents(mask,areaMin=500,areaMax=2500,areaPerimeterRatio=2):
    nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(mask,connectivity = 8)
    sizes = stats[1:,-1]

    a=[]
    finalMask = np.zeros((output.shape))
    for i in range(0,nb_components-1):
        # Get the mask for the i-th contour
        mask_i = np.zeros((output.shape))
        mask_i[ output == i+1 ] = 255
        mask_i = mask_i.astype(np.uint8)

        # Compute the contour
        __,contours, _ = cv2.findContours(mask_i, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter_i = len(contours[0])
        if(sizes[i]> areaMin and sizes[i]<areaMax and sizes[i]/perimeter_i > areaPerimeterRatio):
            a.append(i)
            finalMask[ output == i+1 ] = 255

    finalMask = finalMask.astype(np.uint8)
    return finalMask
