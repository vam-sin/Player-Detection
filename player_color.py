# Even this requires the module lines

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from imageOperatives import *
from ground_color import *

def giveRange(count,threshold,width):
    indices = np.argwhere(count>threshold)
    rangeC = np.array([[0,0]])
    for k in range(0,len(indices)-1):
        left = right = indices[k][0]
        flagL = flagR = False
        for i in range(0,len(count)-1):

            if(indices[k][0]-i > 0 and count[indices[k][0]-i] > width and not flagL):
                left = indices[k][0]-i
            else:
                flagL = True

            if(indices[k][0]+i < len(count) and count[indices[k][0]+i] > width and not flagR):
                right = indices[k][0]+i
            else:
                flagR = True

            if(flagL and flagR):
                rangeC = np.append(rangeC,[[left/256,right/256]],axis=0)
                break
    rangeC = np.delete(rangeC,0,0)
    return rangeC

def segmentsIn(rangeC):
    seg = []
    for i in range(0,len(rangeC)):
        start = round(rangeC[i][0],2)
        end = round(rangeC[i][1],2)
        seg = np.append(seg,[start,end])
        for j in range(1,int(100*(end-start))):
            seg = np.append(seg,round(start+ 0.01*j,2))
    return list(set(seg))

def rangeFor(seg):
    seg.sort()
    rangeC = []
    st = 0
    for i in range(len(seg)-1):
        if(seg[i+1]-seg[i] > 0.015):
            rangeC.append([seg[st]-0.005,seg[i]+0.005])
            st=i+1
    if st!=len(seg)-1 and len(seg)!=0:
        rangeC.append([seg[st]-0.005,seg[i]+0.005])
    return rangeC

def getPlayerColors(stands_mask,groundColorMask,img):

    mask = getGroundObjects(groundColorMask)
    # kernel= np.ones((20,20), np.uint8)
    # mask=mask*stands_mask
    # cv2.imwrite("dmm_o.png", mask*255)
    # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    # cv2.imwrite("dmm.png", mask*255)
    # cv2.imwrite()
    cv2.imwrite('playerMASK.png', mask*255)
    all_players = extractComponents(mask*stands_mask,500,10000,3.5)
    ## mask conains only thet players now without any lines
    st = np.zeros((1080,1920,3))
    st[:,:,0]=st[:,:,1]=st[:,:,2]=all_players/255
    cv2.imwrite("dmm_p.png", st*img)

    nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(all_players,connectivity = 8)
    cv2.imwrite('all_players.png', all_players)
    # print(nb_components,"\n\n\n")
    seg = []
    features = []
    playerIndices = []

    for i in range(0,nb_components-1):
        player = np.zeros((img.shape))

        mask_i = np.zeros((output.shape))
        mask_i[ output == i+1 ] = 1
        mask_i = mask_i.astype(np.uint8)

        # contours, _ = cv2.findContours(mask_i, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # cnt = contours[0]
        # x,y,w,h = cv2.boundingRect(cnt)

        _, contours, _ = cv2.findContours(mask_i, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # cnt = contours[0]
        c = max(contours, key=cv2.contourArea)
        x1 = tuple(c[c[:, :, 0].argmin()][0])[0]
        x2 = tuple(c[c[:, :, 0].argmax()][0])[0]
        y1 = tuple(c[c[:, :, 1].argmin()][0])[1]
        y2 = tuple(c[c[:, :, 1].argmax()][0])[1]
        w = x2-x1
        h = y2-y1
        # print(w,h,"\n\n\n")
        # if(w < 100 and w > 10 and h < 200 and h > 40):

        mask_3i = np.squeeze(np.stack((mask_i,) * 3, -1))
        player = np.multiply(img,mask_3i)
        playerIndices.append(i)

        HSV=rgb_to_hsv(player)
        H=HSV[:,:,0]
        H=H[mask_i==1]
        S=HSV[:,:,1]
        S=S[mask_i==1]
        V=HSV[:,:,2]
        V=V[mask_i==1]

        countH,_ = np.histogram(H,256)
        countS,_ = np.histogram(S,256)
        countV,_ = np.histogram(V,256)

        meanH = np.mean(H)
        meanS = np.mean(S)
        meanV = np.mean(V)
        medianH = np.median(H)
        medianS = np.median(S)
        medianV = np.median(V)

        features.append([meanH,meanS,meanV,medianH,medianS,medianV])

        rangeH = giveRange(countH,15,10)
        rangeS = giveRange(countS,5,4)
        rangeV = giveRange(countV,5,3)

        segH = segmentsIn(rangeH)
        segS = segmentsIn(rangeS)
        segV = segmentsIn(rangeV)

        seg.append([segH,segS,segV])

    print("seg length= ", len(seg))
    #applying Kmeans on features
    features=np.array(features).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _,idx,_ = cv2.kmeans(features,2,None,criteria,15,flags)
    # print(idx)

    #segregating segments based on kmeans
    hsv1=[]
    hsv2=[]
    for i in range(0,3):
        team1 = []
        team2 = []
        for j in range(0,len(idx)):
            if(idx[j]):
                team1 = team1 + seg[j][i]
            else:
                team2 = team2 + seg[j][i]

        hsv1.append(list(set(team1)))
        hsv2.append(list(set(team2)))

    #removing intersections
    sameH = set(hsv1[0]).intersection(hsv2[0])
    setH1 = list(set(hsv1[0])-sameH)
    setH2 = list(set(hsv2[0])-sameH)
    rangeSame = rangeFor(list(sameH))

    # hsv1[0].sort()
    # hsv2[0].sort()
    # print(hsv1[0])
    # print()
    # print(hsv2[0])
    # print()
    # print(rangeSame)
    # print()

    for i in range(0,len(rangeSame)):
        cnt1 = cnt2 = team1 = team2 = 0
        for j in range(0,len(playerIndices)):
            mask_i = np.zeros((output.shape))
            mask_i[ output == playerIndices[j]+1 ] = 1
            mask_i = mask_i.astype(np.uint8)
            mask_3i = np.squeeze(np.stack((mask_i,) * 3, -1))
            player = np.multiply(img,mask_3i)

            HSV=rgb_to_hsv(player)
            H=HSV[:,:,0]

            mask = np.zeros((output.shape))
            mask[ (H > rangeSame[i][0]) &  (H < rangeSame[i][1]) ] = 1
            if(idx[j]):
                team1+=1
                cnt1+=mask.sum()
            else:
                team2+=1
                cnt2+=mask.sum()
        # print(cnt1/team1,cnt2/team2, rangeSame[i])
        # diff = abs(cnt1/team1 - cnt2/team2)
        # if( diff < 20 ):
            # setH1 = np.append(setH1,segmentsIn([rangeSame[i]]))
            # setH2 = np.append(setH2,segmentsIn([rangeSame[i]]))
        if( cnt1/team1 > cnt2/team2 ):
            setH1 = np.append(setH1,segmentsIn([rangeSame[i]]))
        else:
            setH2 = np.append(setH2,segmentsIn([rangeSame[i]]))

    # setH1.sort()
    # setH2.sort()
    # print(setH1)
    # print()
    # print(setH2)
    # print()
    rangeH1 = rangeFor(setH1)
    rangeH2 = rangeFor(setH2)
    rangeS1 = rangeFor(hsv1[1])
    rangeS2 = rangeFor(hsv2[1])
    rangeV1 = rangeFor(hsv1[2])
    rangeV2 = rangeFor(hsv2[2])
    # print("HHHHHH = ", setH1)
    return rangeH1,rangeS1,rangeV1,rangeH2,rangeS2,rangeV2

if __name__ == '__main__' :
    # groundColorMask = cv2.imread("./mask.jpg")
    img = cv2.imread("../Old_Trial/Ball_Detection/frame.jpeg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # getGroundColor function only gets the range of the grass.
    rangeH,rangeS,rangeV = getGroundColor(img)
    # This range of the grass is used to make the mask.
    groundColorMask = rangeToMask(img,[rangeH],[rangeS],[rangeV],0)
    # groundColorMask = rangeToMask(img,[0.22777777777777777, 0.32222222222222224], [0.08235294117647059, 0.4], [0.3843137254901961, 0.6352941176470588],0)
    rangeC = getPlayerColors(groundColorMask,img)
    print(rangeC)
