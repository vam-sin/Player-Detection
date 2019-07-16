import cv2
import numpy as np
from ground_color import getGroundColor, rangeToMask
import time
from scipy import stats
from DLdata import connectDB, disconnectDB, getFrameData, getPlayersMask
from goalLine import getGoalLine
from lines import getLineRange, getStandMask

def getPointsOnMask(stands_mask, playerMask):
    img = stands_mask
    # kernel = np.array([[-1],[1]])
    # dst = cv2.filter2D(img,-1,kernel)
    # # from scipy import ndimage
    # dst = ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    dst = cv2.dilate(img, np.ones((4,4)))   # dont change this
    dst = dst - img
    dst[dst!=0] = 255
    playerMask = cv2.erode(playerMask, np.ones((12,40)))
    dst = np.multiply(dst, playerMask)

    # cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('hmm', 1280,720)
    # cv2.imshow('hmm', playerMask*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('hmm', 1280,720)
    # cv2.imshow('hmm', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    time_init = time.time()
    points = np.array(np.where(dst == 255)).T
    leftPoint = points[np.argmin(points.T[1])]
    rightPoint = points[np.argmax(points.T[1])]
    upperPoints = points[np.where(points.T[0] <= max(leftPoint[0], rightPoint[0]))]
    upperPoints = np.array(sorted(upperPoints, key = lambda x: x[1]))

    return upperPoints, leftPoint, rightPoint


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return np.array([0, 0])

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def score(points, candidatePt, leftPoint, rightPoint):
    tempx = candidatePt[1]
    tempy = candidatePt[0]
    m1 = (tempy-leftPoint[0])/(tempx-leftPoint[1])
    c1 = tempy - m1*tempx
    m2 = (tempy-rightPoint[0])/(tempx-rightPoint[1])
    c2 = tempy - m2*tempx
    leftDist = 0
    rightDist = 0

    pointsL = points[np.where(points.T[1] < candidatePt[1])]
    pointsR = points[np.where(points.T[1] > candidatePt[1])]
    # print(candidatePt, points)

    # print(pointsL, pointsR)

    for pt in pointsL:
        y1 = pt[0]
        y2 = m1*pt[1]+c1
        leftDist += abs(y1-y2)

    for pt in pointsR:
        y1 = pt[0]
        y2 = m2*pt[1]+c2
        rightDist += abs(y1-y2)

    leftDist = leftDist/len(pointsL) if len(pointsL) != 0 else np.inf
    rightDist = rightDist/len(pointsR) if len(pointsR) != 0 else np.inf

    return abs(leftDist), abs(rightDist)


def findUpperLine(points, startingPoint, direction, skip, lrpoints):
    i = 1
    slopes = np.array([])
    pointsContainerX = []
    pointsContainerY = []

    while 1:
        regPoints = points[startingPoint + skip*i*direction : startingPoint + skip*(i+1)*direction: direction]
        # print(len(regPoints))
        # print(startingPoint + skip*i*direction, startingPoint + skip*(i+1)*direction, direction)
        if len(regPoints) == 0:
            break
        y = regPoints.T[0]
        x = regPoints.T[1]
        pointsContainerY.append(y)
        pointsContainerX.append(x)
        slope, intercept, r_value, p_value, std_error = stats.linregress(y, x)
        stdx = np.std(x)
        stdy = np.std(y)
        if(abs(r_value) > 0.85 or (std_error > 1 and (stdy != 0 and ((stdx/stdy > 5) or (stdx/stdy < 1/5))))):
            if std_error > 1 and (stdy != 0 and ((stdx/stdy > 3) or (stdx/stdy < 1/3))):
                slope = (y[-1]-y[0])/(x[-1]-x[0])
            slopes = np.append(slopes, slope)

        i+=1

    histo = np.histogram(slopes, bins=4, range=(-0.4,0.4))
    j = 0
    while 1:
        maxBin = histo[1][histo[0].argsort()[-1-j]]
        selectedSlopes = np.array([])
        pointsY = np.array([])
        pointsX = np.array([])
        prev_i = 0
        for i in range(len(slopes)):
            if slopes[i] >= maxBin and slopes[i] <= maxBin + 0.2:
                if (direction == 1 and np.max(pointsContainerX[i]) <= 1920/2) or (direction == -1 and np.min(pointsContainerX[i]) >= 1920/2):
                    selectedSlopes = np.append(selectedSlopes, slopes[i])
                    pointsY = np.append(pointsY, pointsContainerY[i])
                    pointsX = np.append(pointsX, pointsContainerX[i])

        # pointsX = (np.split(pointsX, np.where(np.diff(pointsX) > 50)[0]+1))
        # pointsY = pointsY[:len(pointsX)]

        if len(selectedSlopes) == 0:
            return np.array([0,0]), np.array([0,0]), np.array([0,0])
        meanX = np.sum(pointsX)/len(pointsX)
        meanY = np.sum(pointsY)/len(pointsY)
        meanSlope = np.sum(selectedSlopes)/len(selectedSlopes)

        if(np.sign(meanSlope) != direction):
            break

        j+=1

    endPoint = np.array([pointsY[-1], pointsX[-1]])
    endPoint = endPoint.astype(np.int_)
    pt1 = np.array([meanY-meanX*meanSlope, 0])
    pt2 = np.array([meanY+meanSlope*(1919-meanX), 1919])
    pt1 = pt1.astype(np.int_)
    pt2 = pt2.astype(np.int_)

    return pt1, pt2, endPoint


def reduceSearchArea(points, nBins = 40):
    points = np.array(points)
    histo = np.histogram(points.T[0], bins=nBins, range=(0,1079))
    maxBin = histo[1][np.where(histo[0] == sorted(histo[0])[-1])][0]
    points = points[np.where((points.T[0] > maxBin-1.5*1079/nBins) & (points.T[0] < maxBin+2.5*1079/nBins))]

    return points


def upperEndPoint(points, line, startingPoint, direction, skip, d_thres = 20):
    def pointLineDist(p1, p2, p3):
        return np.linalg.norm(np.cross(p2-p3, p1-p2))/np.linalg.norm(p2-p3)

    startingPoint = np.where(points == startingPoint)[0][0]
    for point in points[startingPoint::direction*skip]:
        d = pointLineDist(point, *line)
        if d > d_thres:
            endPoint = point
            break

    return point


def sideBoundary(points, cPt):
    y, x = points.T[0], points.T[1]
    slope, intercept, r_value, p_value, std_error = stats.linregress(y, x)

    pt1 = np.array([(0-intercept)/slope, 0])
    pt2 = np.array([(1919-intercept)/slope, 1919])

    pt1, pt2 = pt1.astype(np.int_), pt2.astype(np.int_)
    return [pt1, pt2]


def bottomBoundary(stands_mask):
    if np.sum(stands_mask[-1,:]) == 0:
        bPt1 = [np.nonzero(stands_mask[:,0])[0][-1],0]
        bPt2 = [np.nonzero(stands_mask[:,1919])[0][-1],1919]
        bottomLine = np.array([bPt1, bPt2])

    elif np.sum(stands_mask[-1,:]) > 1700:
        bottomLine = []

    else:
        nonzeros = np.nonzero(stands_mask[-1,:])[0]
        if nonzeros[0] == 0:
            bPt1 = [1079, nonzeros[-1]]
            bPt2 = [np.nonzero(stands_mask[:,-1])[0][-1], 1919]
        else:
            bPt2 = [1079, nonzeros[0]]
            bPt1 = [np.nonzero(stands_mask[:,0])[0][-1], 0]
        bottomLine = np.array([bPt1, bPt2])

    return bottomLine


def checkIfNoCorner(points):
    points = np.array(points)
    leftPoint = points[np.argmin(points.T[1])][0]
    rightPoint = points[np.argmax(points.T[1])][0]

    for i in range(len(points)):
        points[i][0] = points[i][0] + (leftPoint-rightPoint)*i/len(points)
    # print(points)

    stdy = np.std(points.T[0])
    # print(stdy)
    if stdy < 10:
        return 0
    else:
        return 1


# def checkIfNoCorner(center, left, right):
#     m1 = abs((center-left)[0]/(center-left)[1])
#     m2 = abs((center-right)[0]/(center-right)[1])
#     theta = abs(np.arctan(m1)-np.arctan(m2))*180/np.pi
#     print(theta)
#     if theta < 15:
#         return 0
#     else:
#         return 1

def findUpperBounds(points, leftPoint, rightPoint, cornerStatus):
    allPoints = np.array(points)
    # view = 'center'
    if cornerStatus == 1:
        if abs(leftPoint[0] - rightPoint[0]) > 100:
            newPoints = reduceSearchArea(points)
            if leftPoint[0] < rightPoint[0]:
                pt1, pt2, roughEndPoint = findUpperLine(newPoints, 0, 1, 50, (leftPoint, rightPoint))
                if np.linalg.norm(pt1-pt2) == 0:
                    upperLineFound = 0
                else:
                    upperLineFound = 1
                endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], 1, 2)
                points = allPoints[np.where(points.T[1] > endPoint[1])]
                view = 'right'
            else:
                pt1, pt2, roughEndPoint = findUpperLine(newPoints, len(newPoints)-1, -1, 50, (leftPoint, rightPoint))
                if np.linalg.norm(pt1-pt2) == 0:
                    upperLineFound = 0
                else:
                    upperLineFound = 1
                endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], -1, 2)
                points = allPoints[np.where(points.T[1] < endPoint[1])]
                view = 'left'
        else:
            midPt = points[len(points)//2]
            leftScore, rightScore = score(points, midPt, leftPoint, rightPoint)
            newPoints = points
            if leftScore < rightScore:
                pt1, pt2, roughEndPoint = findUpperLine(newPoints, 0, 1, 50, (leftPoint, rightPoint))
                if np.linalg.norm(pt1-pt2) == 0:
                    upperLineFound = 0
                else:
                    upperLineFound = 1
                endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], 1, 2)
                points = allPoints[np.where(points.T[1] > endPoint[1])]
                view = 'right'
            else:
                pt1, pt2, roughEndPoint = findUpperLine(newPoints, len(newPoints)-1, -1, 50, (leftPoint, rightPoint))
                if np.linalg.norm(pt1-pt2) == 0:
                    upperLineFound = 0
                else:
                    upperLineFound = 1
                endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], -1, 2)
                points = allPoints[np.where(points.T[1] < endPoint[1])]
                view = 'left'

        if upperLineFound == 1:
            topLine = np.array([pt1, pt2])
            if len(points) != 0:
                sideLine = sideBoundary(points, endPoint)
            else:
                sideLine = np.array([[0,0], [0,0]])
            cornerPt = line_intersection(sideLine, topLine)
            cornerPt = cornerPt.astype(np.int_)

            # img = np.zeros((1080,1920,3))
            # cv2.line(img, tuple(pt1[::-1]), tuple(pt2[::-1]),(255,0,0),5)
            # cv2.line(img, tuple(sideLine[0][::-1]), tuple(sideLine[1][::-1]),(255,0,0),5)
            # cv2.circle(img, tuple(topLine[0][::-1]), 15, (255,255,0), -15)
            # cv2.circle(img, tuple(sideLine[1][::-1]), 15, (255,255,0), -15)
            # # cv2.circle(img, tuple(endPoint[::-1]), 15, (0,255,0), -15)
            # cv2.circle(img, tuple(cornerPt[::-1]), 15, (255,255,0), -15)
            # cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('hmm', 1280,720)
            # cv2.imshow('hmm', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if view == 'right':
                upperBound = np.array([topLine[0], cornerPt, sideLine[1]])
                scoreValMain, scoreValOther = score(allPoints, cornerPt, topLine[0], sideLine[1])
            else:
                upperBound = np.array([sideLine[0], cornerPt, topLine[1]])
                scoreValOther, scoreValMain = score(allPoints, cornerPt, sideLine[0], topLine[1])

            def slope(pt1, pt2):
                return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

            if np.sign(slope(upperBound[0], upperBound[1])) == np.sign(slope(upperBound[1], upperBound[2])):
                upperLineFound = 0

            # print(view, scoreValMain, scoreValOther)

        if upperLineFound == 0 or scoreValMain > scoreValOther:
            if abs(leftPoint[0] - rightPoint[0]) <= 100:
                midPt = allPoints[len(allPoints)//2]
                leftScore, rightScore = score(allPoints, midPt, leftPoint, rightPoint)
                newPoints = allPoints
                if leftScore > rightScore:
                    pt1, pt2, roughEndPoint = findUpperLine(newPoints, 0, 1, 50, (leftPoint, rightPoint))
                    endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], 1, 2)
                    points = allPoints[np.where(allPoints.T[1] > endPoint[1])]
                    view2 = 'right'
                else:
                    pt1, pt2, roughEndPoint = findUpperLine(newPoints, len(newPoints)-1, -1, 50, (leftPoint, rightPoint))
                    endPoint = upperEndPoint(newPoints, (pt1,pt2), roughEndPoint[1], -1, 2)
                    points = allPoints[np.where(allPoints.T[1] < endPoint[1])]
                    view2 = 'left'

            topLine = np.array([pt1, pt2])
            if len(points) != 0:
                sideLine = sideBoundary(points, endPoint)
            else:
                sideLine = np.array([[0,0], [0,0]])
            cornerPt = line_intersection(sideLine, topLine)
            cornerPt = cornerPt.astype(np.int_)

            if view2 == 'right':
                upperBound2 = np.array([topLine[0], cornerPt, sideLine[1]])
                scoreValMain2, scoreValOther2 = score(allPoints, cornerPt, topLine[0], sideLine[1])
            else:
                upperBound2 = np.array([sideLine[0], cornerPt, topLine[1]])
                scoreValOther2, scoreValMain2 = score(allPoints, cornerPt, sideLine[0], topLine[1])

            # print(view, scoreValMain, scoreValOther)
            # print(view2, scoreValMain2, scoreValOther2)

            upperBound = upperBound if upperLineFound == 1 and scoreValMain < scoreValMain2 else upperBound2
            view = view if upperLineFound == 1 and scoreValMain < scoreValMain2 else view2

        # print(view)

        def absSlope(pt1, pt2):
            return abs((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))

        print(np.arctan(absSlope(upperBound[0], upperBound[1]))*180/np.pi, np.arctan(absSlope(upperBound[1], upperBound[2]))*180/np.pi)
        if abs(np.arctan(absSlope(upperBound[0], upperBound[1]))*180/np.pi - np.arctan(absSlope(upperBound[1], upperBound[2]))*180/np.pi) < 0.5:
            upperBound = np.array([leftPoint, rightPoint])

    else:
        upperBound = np.array([leftPoint, rightPoint])

    return upperBound


def findOuterBoundaries(stands_mask, playerMask):
    points, leftPoint, rightPoint = getPointsOnMask(stands_mask, playerMask)

    cornerStatus = checkIfNoCorner(points)
    if np.sum(stands_mask[0,:]) < 1900:
        upperBound = findUpperBounds(points, leftPoint, rightPoint, cornerStatus)

        if np.any(abs(upperBound) > 1920):
            upperBound = findUpperBounds(points, leftPoint, rightPoint, 0)
    else:
        upperBound = np.array([])

    bottomLine = bottomBoundary(stands_mask)
    if bottomLine != []:
        lowerBound = np.array(bottomLine)
    else:
        lowerBound = np.array([])

    if upperBound != []:
        upperBound = np.fliplr(upperBound).T
    if lowerBound != []:
        lowerBound = np.fliplr(lowerBound).T
    return upperBound, lowerBound


def improveCorner(img, goalData, upperBound, goalPoints):
    if len(upperBound.T) == 3:
        upperBound = np.array(upperBound)
        (top1, top2, bottom1, bottom2) = goalPoints
        bBox = goalData[0]
        [lPt, cPt, rPt] = upperBound.T

        def absSlope(pt1, pt2):
            return abs((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))

        if ((bottom1+bottom2)/2)[0] > cPt[0] and ((bottom1+bottom2)/2)[0] < rPt[0]:
            ub1 = lPt
            ub2 = cPt
        else:
            ub1 = cPt
            ub2 = rPt

        goalLine = [bottom1, bottom2]
        goalLine = sorted(goalLine, key = lambda x : x[0])
        if np.linalg.norm(goalLine[0]-goalLine[1]) == 0:
            status = 0
            return upperBound

        ub1, ub2 = ub1.astype(np.int_), ub2.astype(np.int_)
        interPt = line_intersection((ub1, ub2), tuple(goalLine))

        # blank = np.array(img)
        # cv2.rectangle(img, (bBox[0], bBox[1]), (bBox[2], bBox[3]), (0,255,255), 2)
        # pts = np.array([top1,top2,bottom2,bottom1], np.int32)
        # cv2.fillPoly(blank, pts=[np.array([top1, top2, bottom2, bottom1])], color=(255,0,0))
        # img = cv2.addWeighted(img,0.7,blank,0.3,0)

        if ((bottom1+bottom2)/2)[0] > cPt[0] and ((bottom1+bottom2)/2)[0] < rPt[0]:
            goalLineEdgePt = [1919, goalLine[1][1] + ((goalLine[1][1]-goalLine[0][1])/(goalLine[1][0]-goalLine[0][0]))*(1919-goalLine[1][0])]
            upperBound = np.array([lPt, interPt, goalLineEdgePt]).T
        else:
            goalLineEdgePt = [0, goalLine[0][1] - ((goalLine[1][1]-goalLine[0][1])/(goalLine[1][0]-goalLine[0][0]))*goalLine[0][0]]
            upperBound = np.array([goalLineEdgePt, interPt, rPt]).T

    elif len(upperBound.T) == 2:
        upperBound = np.array(upperBound)
        (top1, top2, bottom1, bottom2) = goalPoints
        bBox = goalData[0]
        [lPt, rPt] = upperBound.T
        goalLine = [bottom1, bottom2]
        goalLine = sorted(goalLine, key = lambda x : x[0])

        if np.linalg.norm(goalLine[0]-goalLine[1]) == 0:
            status = 0
            return upperBound

        leftPt = [0, goalLine[0][1] - ((goalLine[1][1]-goalLine[0][1])/(goalLine[1][0]-goalLine[0][0]))*goalLine[0][0]]
        rightPt = [1919, goalLine[1][1] + ((goalLine[1][1]-goalLine[0][1])/(goalLine[1][0]-goalLine[0][0]))*(1919-goalLine[1][0])]

        upperBound = np.array([leftPt, rightPt]).T

    else:
        pass


    upperBound = upperBound.astype(np.int_)
    return upperBound

if __name__ == '__main__':
    import os
    # os.chdir('../imgs1/')
    img_dir = '../../Videos/imgs1/'
    filename = '291.jpg'
    img = cv2.imread(img_dir+filename, 1)
    frameNum = filename.split('.')[0]
    # rangeH, rangeS, rangeV = getGroundColor(img)
    # print(rangeH, rangeS, rangeV, end='\n')
    # rangeH, rangeS, rangeV = [0.3111111111111111, 0.4388888888888889], [0.3058823529411765, 0.5882352941176471], [0.38823529411764707, 0.5490196078431373] #kc
    rangeH, rangeS, rangeV = [0.3, 0.39444444444444443], [0.3843137254901961, 0.6470588235294118], [0.5137254901960784, 0.7372549019607844]
    # rangeH, rangeS, rangeV = [0.28888888888888886, 0.3888888888888889], [0.3843137254901961, 0.6549019607843137], [0.5176470588235295, 0.7411764705882353] #ck
    ground_mask = rangeToMask(img, [rangeH],[rangeS],[rangeV])
    cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hmm', 1280,720)
    cv2.imshow('hmm', ground_mask*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dbDir = '../DB/data1.db'
    db = connectDB(dbDir)
    playerData, ballData, goalData = getFrameData(db, frameNum)
    disconnectDB(db)
    playerMask = np.array(getPlayersMask(playerData, ballData))
    white_thres = getLineRange(img, ground_mask, playerMask)
    stands_mask = getStandMask(ground_mask, playerMask)
    time_init = time.time()
    upperBound, lowerBound = findOuterBoundaries(stands_mask, playerMask)
    # upperBound, status, img = improveCorner(img, goalData, upperBound, white_thres)
    # print(status)
    print(upperBound, '\n', lowerBound)
    print('Time = ', time.time()-time_init)

    if upperBound != []:
        upperBound = np.fliplr(upperBound.T)
    if lowerBound != []:
        lowerBound = np.fliplr(lowerBound.T)

    if upperBound != []:
        cv2.line(img, tuple(upperBound[0][::-1]),tuple(upperBound[1][::-1]),(255,0,0),5)
        if len(upperBound) > 2:
            cv2.line(img, tuple(upperBound[1][::-1]),tuple(upperBound[2][::-1]),(255,0,0),5)
            cv2.circle(img, tuple(upperBound[1][::-1]), 10, (0,0,255), -10)
    if lowerBound != []:
        cv2.line(img, tuple(lowerBound[0][::-1]),tuple(lowerBound[1][::-1]),(255,0,0),5)

    cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hmm', 1280,720)
    cv2.imshow('hmm', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = np.squeeze(np.stack((stands_mask*255,) * 3, -1))
    if upperBound != []:
        cv2.line(img, tuple(upperBound[0][::-1]),tuple(upperBound[1][::-1]),(255,0,0),5)
        if len(upperBound) > 2:
            cv2.line(img, tuple(upperBound[1][::-1]),tuple(upperBound[2][::-1]),(255,0,0),5)
            cv2.circle(img, tuple(upperBound[1][::-1]), 10, (0,0,255), -10)
    if lowerBound != []:
        cv2.line(img, tuple(lowerBound[0][::-1]),tuple(lowerBound[1][::-1]),(255,0,0),5)

    cv2.namedWindow('hmm',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hmm', 1280,720)
    cv2.imshow('hmm', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # import os
    # folder = '../imgsGoal/'
    # dbDir = '../DB/data1.db'
    # db = connectDB(dbDir)
    # for filename in sorted(os.listdir(folder)):
    #     img = cv2.imread(os.path.join(folder,filename))
    #     if img is not None:
    #         print(filename)
    #         frameNum = filename.split('.')[0]
    #         rangeH, rangeS, rangeV = [0.28888888888888886, 0.3888888888888889], [0.3843137254901961, 0.6549019607843137], [0.5176470588235295, 0.7411764705882353]
    #         ground_mask = rangeToMask(img, [rangeH],[rangeS],[rangeV], 0.03, 0.03)
    #         playerData, ballData, goalData = getFrameData(db, frameNum)
    #         playerMask = np.array(getPlayersMask(playerData, ballData))
    #         white_thres = getLineRange(img, ground_mask, playerMask)
    #         # cv2.imshow('hmm', ground_mask*255)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         stands_mask = getStandMask(ground_mask, playerMask)
    #         # time_init = time.time()
    #         # try:
    #         upperBound, lowerBound = findOuterBoundaries(stands_mask, playerMask)
    #         upperBound, status, img = improveCorner(img, goalData, upperBound, white_thres)
    #         # except:
    #         #     print("Error")
    #         #     upperBound, lowerBound = [], []
    #         # print('Time = ', time.time()-time_init)
    #         # print(upperBound, lowerBound)
    #
    #         if upperBound != []:
    #             upperBound = np.fliplr(upperBound.T)
    #         if lowerBound != []:
    #             lowerBound = np.fliplr(lowerBound.T)
    #
    #         if upperBound != []:
    #             cv2.line(img, tuple(upperBound[0][::-1]),tuple(upperBound[1][::-1]),(255,0,0),5)
    #             if len(upperBound) > 2:
    #                 cv2.line(img, tuple(upperBound[1][::-1]),tuple(upperBound[2][::-1]),(255,0,0),5)
    #                 cv2.circle(img, tuple(upperBound[1][::-1]), 10, (0,0,255), -10)
    #         if lowerBound != []:
    #             cv2.line(img, tuple(lowerBound[0][::-1]),tuple(lowerBound[1][::-1]),(255,0,0),5)
    #         # cv2.imshow('hmm', img)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         cv2.imwrite(os.path.join(folder,filename), img)
    # disconnectDB(db)
