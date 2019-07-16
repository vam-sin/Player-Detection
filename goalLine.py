import cv2
import numpy as np
import time
from DLdata import connectDB, disconnectDB, getFrameData, getPlayersMask
from lines import getLineRange
from ground_color import *


def consecutiveSplits(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)


def getThreshold(image, window_size):
    thres = 250
    saveDict = {}
    while thres > 200:
        img = np.zeros(image.shape)
        img[image > thres] = 255

        total_boxes = 0
        green_boxes = 0

        for i in range(int(img.shape[0]/window_size)):
            for j in range(int(img.shape[1]/window_size)):
                # break
                edge = window_size
                window = img[i*edge:(i+1)*edge, j*edge:(j+1)*edge]
                area = np.sum(window)/255
                total_boxes += 1
                if(area > 20):
                    points = np.where(window == 255)
                    x = points[0]
                    y = points[1]
                    stdx = np.std(x)
                    stdy = np.std(y)
                    if(stdy != 0 and stdx/stdy < 1/3):
                        # cv2.rectangle(img, (j*edge, i*edge), ((j+1)*edge, (i+1)*edge), (0,255,0),1)
                        green_boxes += 1

        saveDict[thres] = green_boxes

        thres -= 2

    values = list(saveDict.values())
    return list(saveDict.keys())[values.index(max(values))]


def findPoleBottom(img, stride=10):
    def getPoleData(img):
        compressed = np.average(img, axis=1)
        compressed[compressed != 0] = 1
        if np.sum(compressed) > img.shape[0]//2:
            pole = max(consecutiveSplits(np.nonzero(compressed)[0]), key=len)
            poleLength = len(pole)
            poleBottom = pole[-1]
            return poleBottom, poleLength
        else:
            return 0, 0

    i = 0
    maxLength = 0
    while 1:
        crop = img[:, i*stride:(i+1)*stride]
        bottomPt, length = getPoleData(crop)
        if length > maxLength:
            maxLength = length
            poleBottom = np.array([int((i+0.5)*stride), bottomPt])
        i += 1
        if (i+1)*stride >= img.shape[1]:
            break
    if maxLength != 0:
        return poleBottom
    else:
        return np.array([0,0])


def getPoleLength(img, topPoint):
    # cv2.imshow('hmm', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    compressed = np.average(img, axis=1)
    compressed[compressed != 0] = 1
    compressed[:topPoint] = 0
    poleLength = len(max(consecutiveSplits(np.nonzero(compressed)[0]), key=len))

    return poleLength


def getTopBar(image, window_size, thetaRange=[0,30], minArea=40):
    # cv2.imshow('hmm', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = image[:image.shape[0]//2, :]
    leftPoints = []
    rightPoints = []
    for i in range(int(img.shape[0]/window_size)):
        for j in range(int(img.shape[1]/window_size)):
            edge = window_size
            window = img[i*edge:(i+1)*edge, j*edge:(j+1)*edge]
            area = np.sum(window)/255
            if area > minArea and area < window_size**2//3:
                points = np.array(np.where(window == 255))
                points = (points.T[points[1,:].argsort()]).T    #sort points based on horizontal coordinate
                x = points[0]
                y = points[1]
                x1 = np.average(x[np.where(y == y[0])])
                x2 = np.average(x[np.where(y == y[-1])])
                y1, y2 = y[0], y[-1]
                slope = (x2-x1)/(y2-y1)
                theta = np.arctan(slope)*180/np.pi
                if(theta > thetaRange[0] and theta < thetaRange[1]):
                    # print(thetaRange, theta)
                    blank = np.zeros(img.shape)
                    pt1 = np.array([i*edge+x1, j*edge+y1])
                    pt2 = np.array([i*edge+x2, j*edge+y2])
                    pt1Edge = np.array([pt1[0]-pt1[1]*(pt2[0]-pt1[0])/(pt2[1]-pt1[1]) , 0])
                    pt2Edge = np.array([pt2[0]+(img.shape[1]-pt1[1])*(pt2[0]-pt1[0])/(pt2[1]-pt1[1]) , img.shape[1]])
                    pt1Edge = pt1Edge.astype(np.int_)
                    pt2Edge = pt2Edge.astype(np.int_)
                    cv2.line(blank, tuple(pt1Edge[::-1]), tuple(pt2Edge[::-1]), 1, 5)
                    blank = np.multiply(blank, img)/255
                    # cv2.imshow('hmm', blank)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    if np.sum(blank) > img.shape[1]:
                        points = np.array(np.where(blank == 1))
                        splits = consecutiveSplits(sorted(points[1]), 5)
                        biggestLine = sorted(splits, key = len)[-1]
                        blank[:, :np.min(biggestLine)] = 0
                        blank[:, np.max(biggestLine):] = 0
                        points = np.array(np.where(blank == 1))
                        points = (points.T[points[1,:].argsort()]).T
                        x = points[0]
                        y = points[1]
                        x1 = np.average(x[np.where(y == y[0])])
                        x2 = np.average(x[np.where(y == y[-1])])
                        y1, y2 = y[0], y[-1]
                        leftPoint, rightPoint = np.array([x1, y1]), np.array([x2, y2])
                        # print(leftPoint, rightPoint)
                        leftPoints.append(leftPoint)
                        rightPoints.append(rightPoint)

    leftPoints, rightPoints = np.array(leftPoints), np.array(rightPoints)
    if len(leftPoints != 0):
        leftAvg = np.average(leftPoints, axis=0)[0]
        rightAvg = np.average(rightPoints, axis=0)[0]
        leftPoints = np.array([x for x in leftPoints if x[0] < leftAvg+img.shape[1]/30])
        rightPoints = np.array([x for x in rightPoints if x[0] < rightAvg+img.shape[1]/30])
        leftPoint = leftPoints[leftPoints[:,1].argsort()][0]
        rightPoint = rightPoints[rightPoints[:,1].argsort()][-1]
        # print(leftPoint, rightPoint)
        leftPoint, rightPoint = leftPoint.astype(np.int_)[::-1], rightPoint.astype(np.int_)[::-1]
    else:
        leftPoint, rightPoint = np.array([0,0]), np.array([0,0])

    return leftPoint, rightPoint


def getGoalLine(img, bBox, line_thres, view, goalRatios = [2.1016, 2.0222], perspAdjustRatios = [1.0787, 0.9270], goalLineAngleRanges = [[0, 30], [-30,0]], bBoxTolerance = 20):
    bBox[0] = bBox[0] - bBoxTolerance if bBox[0] - bBoxTolerance > 0 else bBox[0]
    bBox[1] = bBox[1] - bBoxTolerance if bBox[1] - bBoxTolerance > 0 else bBox[1]
    bBox[2] = bBox[2] + bBoxTolerance if bBox[2] - bBoxTolerance < 1920 else bBox[2]
    bBox[3] = bBox[3] + bBoxTolerance if bBox[3] - bBoxTolerance < 1080 else bBox[3]

    bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cropImgColor = np.array(bw_img[bBox[1]:bBox[3], bBox[0]:bBox[2]])

    if view == 'left':
        goalLineAngleRange = goalLineAngleRanges[0]
        goalRatio = goalRatios[0]
        perspAdjustRatio = perspAdjustRatios[0]
    else:
        goalLineAngleRange = goalLineAngleRanges[1]
        goalRatio = goalRatios[1]
        perspAdjustRatio = perspAdjustRatios[1]

    window_size = cropImgColor.shape[1]//10
    thres = getThreshold(cropImgColor, window_size)
    cropImg = np.zeros(cropImgColor.shape)
    cropImg[cropImgColor >= thres] = 255
    # cv2.imshow('hmm', cropImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    top1, top2 = getTopBar(cropImg, window_size, goalLineAngleRange)

    allLinesImg = np.zeros(cropImgColor.shape)
    allLinesImg[cropImgColor > line_thres] = 1
    # cv2.imshow('hmm', allLinesImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    poleLen1 = getPoleLength(allLinesImg[:,max(0, top1[0]-7): min(top1[0]+7, allLinesImg.shape[1])], top1[1])
    poleLen2 = getPoleLength(allLinesImg[:,max(0, top2[0]-7): min(top2[0]+7, allLinesImg.shape[1])], top2[1])
    print(allLinesImg.shape[0]-2*bBoxTolerance)
    poleLen1 =  poleLen1 if poleLen1 > (allLinesImg.shape[0]-2*bBoxTolerance)*0.66 and poleLen1 < (allLinesImg.shape[0]-2*bBoxTolerance)*0.9 else -1
    poleLen2 =  poleLen2 if poleLen2 > (allLinesImg.shape[0]-2*bBoxTolerance)*0.66 and poleLen2 < (allLinesImg.shape[0]-2*bBoxTolerance)*0.9 else -1
    print(poleLen1, poleLen2)

    def polyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def absAngle(pt1, pt2):
        return np.arctan(abs((pt2[1]-pt1[1])/(pt2[0]-pt1[0])))*180/np.pi

    if np.linalg.norm(top1-top2) != 0:
        top1, top2 = top1 + bBox[:2], top2 + bBox[:2]
        topLength = np.linalg.norm(top1-top2)
        hypotheticalLength = topLength/goalRatio
        # +/-2 for perspective adjustment
        if poleLen1 == -1 or poleLen2 == -1:
            if poleLen1 != -1:
                poleLen2 = poleLen1 / perspAdjustRatio
            elif poleLen2 != -1:
                poleLen1 = poleLen2 * perspAdjustRatio
            else:
                poleLen1 = hypotheticalLength
                poleLen2 = hypotheticalLength / perspAdjustRatio
        # print(poleLen1, poleLen2)
        bottom1 = np.array(top1)
        bottom1[1] = bottom1[1] + poleLen1
        bottom2 = np.array(top2)
        bottom2[1] = bottom2[1] + poleLen2

        [x, y] = np.array([top1, top2, bottom2, bottom1]).T
        if polyArea(x, y) > (cropImg.shape[0]-bBoxTolerance*2)*(cropImg.shape[1]-bBoxTolerance*2)//2:
            goalFound = 1
        else:
            goalFound = 0

    else:
        goalFound = 0

    if goalFound == 1:
        return top1, top2, bottom1, bottom2
    else:
        bottom1 = findPoleBottom(allLinesImg[:,:allLinesImg.shape[1]//2])
        bottom2 = findPoleBottom(allLinesImg[:,allLinesImg.shape[1]//2:])
        bottom1, bottom2 = bottom1 + bBox[:2], bottom2 + bBox[:2] + np.array([allLinesImg.shape[1]//2, 0])
        bottomLength = np.linalg.norm(bottom1-bottom2)
        hypotheticalLength = bottomLength/goalRatio
        poleLen1 = hypotheticalLength
        poleLen2 = hypotheticalLength / perspAdjustRatio
        top1 = np.array(bottom1)
        top1[1] = top1[1] - poleLen1
        top2 = np.array(bottom2)
        top2[1] = top2[1] - poleLen2
        [x, y] = np.array([top1, top2, bottom2, bottom1]).T
        if polyArea(x, y) > (cropImg.shape[0]-bBoxTolerance*2)*(cropImg.shape[1]-bBoxTolerance*2)//2:
            return top1, top2, bottom1, bottom2
        else:
            return np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])


if __name__ == '__main__':
    imgDir = '../'
    filename = '169.jpg'
    img = cv2.imread(imgDir+filename, 1)
    frameNum = filename.split('.')[0]
    dbDir = '../DB/data1.db'
    db = connectDB(dbDir)
    playerData, ballData, goalData = getFrameData(db, frameNum)
    disconnectDB(db)
    bBox = goalData[0]

    import pickle
    f = open('goalData.dat', 'rb')
    goalDict = pickle.load(f)
    f.close()
    goalData = goalDict[1]
    # rangeH, rangeS, rangeV = getGroundColor(img)
    # print(rangeH, rangeS, rangeV)
    # rangeH, rangeS, rangeV = [0.28888888888888886, 0.3888888888888889], [0.3843137254901961, 0.6549019607843137], [0.5176470588235295, 0.7411764705882353]
    rangeH, rangeS, rangeV = [0.2777777777777778, 0.36666666666666664], [0.28627450980392155, 0.6235294117647059], [0.22745098039215686, 0.592156862745098]
    ground_mask = rangeToMask(img, [rangeH],[rangeS],[rangeV])
    playerMask = np.array(getPlayersMask(playerData, ballData))
    white_thres = getLineRange(img, ground_mask, playerMask)
    # print(white_thres)
    # cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('hmm', 1280,720)
    # cv2.imshow('hmm', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    time_init = time.time()

    top1, top2, bottom1, bottom2 = getGoalLine(img, bBox, white_thres, 'right', [goalData[0][0][0], goalData[0][1][0]], [goalData[1][0][0], goalData[1][1][0]], goalData[2])
    print('Time = ', time.time()-time_init)
    if np.linalg.norm(bottom1-bottom2) != 0:
        cv2.circle(img, (bottom1[0], bottom1[1]), 3, (255,0,0), -3)
        cv2.circle(img, (bottom2[0], bottom2[1]), 3, (255,0,0), -3)
        cv2.circle(img, (top1[0], top1[1]), 3, (255,0,0), -3)
        cv2.circle(img, (top2[0], top2[1]), 3, (255,0,0), -3)

        blank = np.array(img)
        pts = np.array([top1,top2,bottom2,bottom1], np.int32)
        cv2.fillPoly(blank, pts=[pts], color=(255,0,0))
        img = cv2.addWeighted(img,0.7,blank,0.3,0)

        # slope = (bottom2[1]-bottom1[1]) / (bottom1[0]-bottom2[0])
        # cv2.line(img,(bottom1[0]-1500,int(bottom1[1]+1500*slope)),(bottom2[0]+1500,int(bottom2[1]-1500*slope)),(255,0,0),5)
        cv2.imshow('hmm', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
