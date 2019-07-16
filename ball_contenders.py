import math
import numpy as np
import cv2
import sqlite3
import os
import io
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(out.read())
    return np.load(out)


def form_box(lin_p,lines,stats,j,thres_val,area_o, in_player):

    lineThickness=1
    print('stats= ', stats)
    a=stats[0]
    b=stats[1]
    c=stats[2]
    d=stats[3]
    #removing all those bounding boxes which aren't approximately squares
    if(abs(c-d)>2):
        in_player[j]=0
        return
    #removing all those bounding boxes whose white area is <= 60% of total area of bounding box
    if(c*d*0.6>stats[-1]):
        in_player[j]=0
        return
    tuple_of_colors=(0,0,0)
    #increasing box's height and width by 2 pixels
    if(a>2):
        a1=a-2
    else:
        a1=a
    if(b>2):
        b1=b-2
    else:
        b1=b
    if(a1+c+2<1920):
        c1=c+2
    else:
        c1=c
    if(b1+d+2<1080):
        d1=d+2
    else:
        d1=d
    # mask_p = lin_p[b1:b1+d1, a1:a1+c1,:]
    # gray = cv2.cvtColor(mask_p, cv2.COLOR_BGR2GRAY)
    # sec_ball_prev = lin_p[b:d, a:c, :]
    # sec_ball_af = lin_p[b1:d1, a1:c1, :]
    # print(np.shape(lin_p))
    # print(np.shape(lines))
    lin_pp = lin_p[b:d+b, a:c+a, :]
    lines_pp = lines[b:d+b, a:c+a]
    print("lin_pp:",[a,b,c,d])
    print("lin.shape: ", np.shape(lin_p) )
    cv2.imwrite('org_img_S.png', lin_pp)
    cv2.imwrite('lines_c_S.png', lines_pp*255)
    # cv2.imwrite('sec_ball_prev.png', lin_p)
    # cv2.imwrite('sec_ball_af.png', sec_ball_af)

    area_p = np.sum(np.sum(lines_pp,axis=1),axis=0)
    #checking if increased box has >= 20% increment in white area
    print('areas=',[area_p, area_o])
    if((area_p-area_o)/area_o >= 0.2):
        in_player[j]=0
        return
    cv2.line(lin_p, (a, b), (a+c, b), tuple_of_colors, lineThickness)
    cv2.line(lin_p, (a, b), (a, b+d), tuple_of_colors, lineThickness)
    cv2.line(lin_p, (a, b+d), (a+c, b+d), tuple_of_colors, lineThickness)
    cv2.line(lin_p, (a+c, b), (a+c, b+d), tuple_of_colors, lineThickness)
    return
def get_ball_boxes(org_img, lines):
    # After applying white range on image we get lines.png
    lines[lines>0]=255

    #mask the area in which you do not wanna search for ball
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    lines_p=cv2.morphologyEx(lines, cv2.MORPH_OPEN, kernel)
    lines_p=cv2.morphologyEx(lines_p, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("lineMaskBall.png", lines_p)
    # lines_p is the binary image after opening/closing lines.png
    # cv2.imwrite('lines_p.png', lines_p)
    lines_p=lines_p/255
    lines_p = np.array(lines_p, dtype=np.uint8)

    # cv2.imwrite('org_img_S.png', org_img)
    # cv2.imwrite('lines_c_S.png', lines_p)

    nb_comp, output, stats, centroids = cv2.connectedComponentsWithStats(lines_p, connectivity=8)

    peri = np.empty([nb_comp-1,2])
    cir =  np.empty([nb_comp-1,2])

    area_o=[]
    for i in range(1,nb_comp):
        maskp = lines_p[stats[i][1]:stats[i][1]+stats[i][3], stats[i][0]:stats[i][0]+stats[i][2]]
        #finding perimeter of ith connected components
        __,	contoursp,_ = cv2.findContours(maskp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        con = contoursp[0]
        per=len(con)
        # print([i,per])
        for j in range(len(con)-1):
            if(con[j][0][0]==con[j+1][0][0]):
                per+=abs(con[j+1][0][1]-con[j][0][1])-1
            elif(con[j][0][1]==con[j+1][0][1]):
                per+=abs(con[j+1][0][0]-con[j][0][0])-1
        if(con[0][0][0]==con[-1][0][0]):
            per+= abs(con[0][0][1]-con[-1][0][1])-1
        elif(con[0][0][1]==con[-1][0][1]):
            per+= abs(con[0][0][0]-con[-1][0][0])-1
        peri[i-1]=[per,i]
        ar =per**2/ (4*(math.pi)*stats[i][-1])
        cir[i-1]=[ar,i]
        area_o.append(np.sum(np.sum(maskp,axis=1),axis=0))
    # org_img is the original image

    in_player=[1 for k in range(nb_comp-1)]

    #iterate through all bounding boxes of players and make in_player of that ball
    # candidate=0 if it lies in any of them
    print('nbcomp= ',nb_comp)
    for i in range(nb_comp-1):

        if(in_player[i]==1 and cir[i][0]<=1.01 and cir[i][0]>=0.6):
            j = int(cir[i][1])
            print([i, cir[i], stats[j], area_o[i]])
            form_box(org_img,lines_p,stats[j],i,0,area_o[i], in_player)
        else:
            j = int(cir[i][1])
            in_player[i]=0
    cv2.imwrite('lino_p.png', org_img)
    lis_stats = []
    for i in range(nb_comp-1):
        if(in_player[i]==1):
            lis_stats.append([stats[i+1][-1], stats[i+1]])
    lis_stats.sort(reverse=True)
    # print('lis_stats: ', lis_stats[0][1])
    return lis_stats

def getdata(query,id):
    # print(query)
    DATABASE = "/home/ancalagon/data"
    conn = sqlite3.connect(DATABASE+str(id)+'.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(query)
    rows=c.fetchall()
    conn.close()
    return rows[0]
# def insert(id, rangeH, rangeS, rangeV, whiteL, whiteC, whiteR,status,img,cas, c):
#     qu = 'INSERT INTO color VALUES(?,?,?,?,?,?,?,?,?,?)'
#     x = tuple([id, rangeH, rangeS, rangeV, whiteL, whiteC, whiteR,status,cas,img])
#     c.execute(qu, x)
# def update(ball,fid, i):
#     conn = sqlite3.connect(DATABASE+str(i)+'.db', detect_types=sqlite3.PARSE_DECLTYPES)
#     c = conn.cursor()
#     query = 'UPDATE tab SET ball=? WHERE frameID='+fid
#     c.execute(query, tuple(ball))
#     conn.commit()
#     conn.close()
#if in_player[i]=1 implies connected comp. i+1 is contender for the ball
'''
Two tasks left:
#1 Remove all boxes which lie inside the bounding box
#2 If at the end there are 2 or more contenders for ball
   output the one with max area
'''
def update_ball(org_img, lines, idp, fid):
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("arr", convert_array)
    query = "SELECT id, ball FROM tab WHERE frameID="+fid
    # print([fid, idp])
    row= getdata(query, idp)
    ids, balls = row[0], row[1]
    # print(balls)
    if(len(balls))==0:
        print('typoooooo = ', type(fid))
        queryp = "SELECT ball FROM tab WHERE frameID="+str(int(fid)-1)
        balls = getdata(queryp, idp)
        if(len(balls)==0):
            return


    for k in range(len(balls)):
        ball = balls[k]
        a = max(ball[0]-80, 0)
        c = min(ball[2]+80, 1920)
        b = max(ball[1]-80, 0)
        d = min(ball[3]+80, 1080)
        org_img_c = org_img[b:d,a:c,:]
        lines_c = lines[b:d,a:c]
        list_possible = get_ball_boxes(org_img_c, lines_c)

        if(len(list_possible)==0):
            continue
        else:
            ball_stat= list_possible[0][1]
            ball = np.asarray([ball_stat[0]+a, ball_stat[1]+b, ball_stat[2]+ball_stat[0]+a, ball_stat[3]+ball_stat[1]+b])
            # print('ball= ', ball)
            query = 'UPDATE tab SET ball=? WHERE frameID='+fid
            # print('query and ball: ', [query, ball])
            return query, ball
    return None, balls
# if __name__ == '__main__':
