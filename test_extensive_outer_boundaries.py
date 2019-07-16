from multiprocessing import Process, current_process, Queue
from ground_color import *
from player_color import *
from find_outline_boundaries import *
from lines import *
import pickle
import os
import sys
from find_inner_boundaries import *
from line_merge import *
from end_points import *
from ball_contenders import update_ball, get_ball_boxes, form_box
from goalLine import *
import sqlite3
import os
import io
DATABASE = "/home/ancalagon/data"
file_Name = "/home/ancalagon/Pickle_File_color_data"

fileObject = open(file_Name,'rb')
# load the object from the file into var b
color_ranges = pickle.load(fileObject)
fileObject.close()

f = open('goalData.dat', 'rb')
goalDict = pickle.load(f)
f.close()

ground_color_range=color_ranges['ground_color']
line_color_range=color_ranges['Line color']
print(ground_color_range)
font = cv2.FONT_HERSHEY_SIMPLEX

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

def error_in_ground_mask(ground_mask, playerMask):

	kernel = np.ones((15,15),np.uint8)
	ground_mask = cv2.dilate(ground_mask, kernel)
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
	cv2.imwrite('before_error.png', stands_mask*255)
	kernel = np.ones((80,80),np.uint8)
	stands_mask = cv2.morphologyEx(stands_mask, cv2.MORPH_OPEN, kernel)
	cv2.imwrite('before_erroro.png', stands_mask*255)
	# initial_e = np.sum(np.sum(stands_mask,axis=1),axis=0)
	kernel = np.ones((15,15),np.uint8)
	kernel_p = np.ones((8,8), np.uint8)
	stands_mas = cv2.morphologyEx(stands_mask, cv2.MORPH_CLOSE, kernel)
	# stands_ma = cv2.morphologyEx(stands_mas, cv2.MORPH_OPEN, kernel_p)
	# cv2.imwrite('mid_error.png', stands_mas*255)
	cv2.imwrite('after_error.png', stands_mas*255)
	delta = stands_mas-stands_mask
	delta = cv2.morphologyEx(delta, cv2.MORPH_OPEN, kernel_p)
	delta = np.multiply(delta,playerMask)
	cv2.imwrite('delta.png', delta*255)
	err_points = np.where(delta==1)
	error = np.sum(np.sum(delta,axis=1),axis=0)
	print("error = ", error)
	return error, err_points

def Draw_Upper_Lower(img,groundColorMask,stands_mask,dir_name,file_name,upperBound,lowerBound):
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
	temp_name=dir_name+"/"+file_name
	img[:,:,0]=img[:,:,0]+stands_mask*127
	img[:,:,1]=img[:,:,1]
	cv2.imwrite('gmask_p.png', groundColorMask*255)
	cv2.imwrite('gmask_p.png', stands_mask*255)
	img[:,:,2]=img[:,:,2]-groundColorMask*50

	cv2.imwrite(temp_name,img)

def Draw_cases_lines(dir_name,file_name,case,para_line,per_line,lineMask,coords):

	temp_name=dir_name+"/"+file_name
	img=cv2.imread(temp_name)
	# print(para_line)
	if(len(para_line)==4):
		cv2.line(img,(int(para_line[0]),int(para_line[1])-5),(int(para_line[2]),int(para_line[3])-5),(255,255,255),5)
		print_x=int((int(para_line[0])+int(para_line[2]))/2)
		print_y=int((int(para_line[1])+int(para_line[3]))/2)-20
		# print(print_x,print_y)
		cv2.putText(img,'para',(print_x,print_y), font, 2,(0,0,255),2,cv2.LINE_AA)
	if(len(per_line)==4):
		cv2.line(img,(int(per_line[0])-5,int(per_line[1])),(int(per_line[2])-5,int(per_line[3])),(255,255,255),5)
		print_x=int((int(per_line[0])+int(per_line[2]))/2)-20
		print_y=int((int(per_line[1])+int(per_line[3]))/2)
		# print(print_x,print_y)

		cv2.putText(img,'per',(print_x,print_y), font, 2,(0,0,255),2,cv2.LINE_AA)

	cv2.putText(img,"case: "+str(case)+" ",(100,100),font,1,(255,255,255),2,cv2.LINE_AA)
	temp_points=np.where(lineMask==255)
	temp_r_real=img[:,:,0]
	temp_r_real[temp_points]=255
	img[:,:,0]=temp_r_real


	temp_r_real=img[:,:,1]
	temp_r_real[temp_points]=255
	img[:,:,1]=temp_r_real

	temp_r_real=img[:,:,2]
	temp_r_real[temp_points]=255
	img[:,:,2]=temp_r_real


	coords=coords.astype('int')
	print(coords)
	for cd in coords:
		cv2.line(img,(cd[0][1],cd[0][0]),(cd[1][1],cd[1][0]),(0,0,255),3)
	cv2.imwrite(temp_name,img)



def legit_frames(player_boxes, upper_bound, lower_bound):
    if len(player_boxes)<=5 or (len(upper_bound)<2 and len(lower_bound)<2):
        return 0
    else:
        for i in range(len(player_boxes)):
            if (player_boxes[i][2]-player_boxes[i][0])*(player_boxes[i][3]-player_boxes[i][1]) > 41472:
                return 0
        return 1
def getdata(query,id):
	conn = sqlite3.connect(DATABASE+str(id)+'.db', detect_types=sqlite3.PARSE_DECLTYPES)
	c = conn.cursor()
	c.execute(query)
	rows=c.fetchall()
	conn.close()

	return rows[0]
def insert(id, rangeH, rangeS, rangeV, whiteL, whiteC, whiteR,status,img,cas, c):
    qu = 'INSERT INTO color VALUES(?,?,?,?,?,?,?,?,?,?)'
    x = tuple([id, rangeH, rangeS, rangeV, whiteL, whiteC, whiteR,status,cas,img])
    c.execute(qu, x)
def update(ball,fid, i):
	conn = sqlite3.connect(DATABASE+str(i)+'.db', detect_types=sqlite3.PARSE_DECLTYPES)
	c = conn.cursor()
	query = 'UPDATE tab SET ball=? WHERE frameID='+fid
	c.execute(query, tuple(ball))
	conn.commit()
	conn.close()

def error_in_linemask(lineMask):
	kernel = np.ones((20,20),np.uint8)
	linemask = cv2.morphologyEx(lineMask, cv2.MORPH_OPEN, kernel)
	cv2.imwrite('linemask.png', linemask)
	return np.sum(np.sum(linemask,axis=1),axis=0)
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
    # print(unique)
    new=np.zeros(maxval)
    for i in range(len(unique)):
        new[unique[i]] = counts[i]

    return np.array(new)
def print_histogram( items ):
    i = 0
    for n in items:
        output = ''
        times = n

        print(i, end='')

        # print(i, end='')

        i+=1
        while( times > 0 ):
          output += '*'
          times = times - 1
        print(output)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("arr", convert_array)

Master_test_dir="./../videos"
Sub_test_dir=sorted(os.listdir(Master_test_dir))

Master_save_outer_boundaries="./Save_outer_boundaries"
os.system("rm -r "+Master_save_outer_boundaries)
os.system("mkdir "+Master_save_outer_boundaries)
# iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", 2)
# print([iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_])
# iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", 2)
# print([iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_])
# conn = sqlite3.connect(DATABASE+str(2)+'.db', detect_types=sqlite3.PARSE_DECLTYPES)
# c = conn.cursor()
# insert(iota+1, rangeH,rangeS,rangeV, whiteL, whiteC, whiteR,1,str(242),c)

# # print(getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", file_test_number))
# conn.commit()
# conn.close()
# iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", 2)
# print([iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_])
alpha_f = 0
beta_f=0
buffer_queries = []
for file_test_number in range(2,3):
    flag = 0
    dir_test_number=int(Sub_test_dir[file_test_number][4:])-1
    # rangeH=ground_color_range[dir_test_number][0]
    # rangeS=ground_color_range[dir_test_number][1]
    # rangeV=ground_color_range[dir_test_number][2]
    # thresL=line_color_range[dir_test_number][0]
    # thresC=line_color_range[dir_test_number][1]
    # thresR=line_color_range[dir_test_number][2]
    # print(rangeH,rangeS,rangeV,"ground_color_ranges\n")
    # print(thresL,thresC,thresR,"line_color_ranges\n")


    Sub_outerboundaries_save_Dir=Master_save_outer_boundaries+"/"+Sub_test_dir[file_test_number]
    # print(Sub_test_dir[file_test_number])

    os.system("mkdir "+Sub_outerboundaries_save_Dir)

    # go_for = multiprocessing.Queue()
    Current_Dir=Master_test_dir+"/"+Sub_test_dir[file_test_number]
    iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,cas,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", file_test_number)
    # print([rangeH, rangeS, rangeV, whiteL, whiteC, whiteR])
    # rangeH = np.asarray([0.22, rangeH[1]])
    print(rangeH)
    # for files_jpg in os.listdir(Current_Dir):
    for x in range(1):
    # print(files_jpg)
        files_jpg = "8.jpg"
        file_name_jpg=Current_Dir+"/"+files_jpg	# for process in processes:
        # 	process.join()les_jpg
        if(files_jpg[-4:]=='.jpg'):
            if(1):
                img = cv2.imread(file_name_jpg)
                # process = Process(target=func, args=(img,))
                # processes.append(process)
                # process.start()
                # iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,cas,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", file_test_number)
                # print([iota,rangeH, rangeS, /rangeV, whiteL, whiteC, whiteR, status,_])

                db = connectDB(DATABASE + str(file_test_number)+'.db')
                playerData, ballData, goalData = getFrameData(db, int(files_jpg[:-4]))
                disconnectDB(db)
                playerMask = np.array(getPlayersMask(playerData, []))

                time_init = time.time()
                groundColorMask = rangeToMask(img,[rangeH],[rangeS],[rangeV])
                cv2.imwrite('gmask.png', groundColorMask*255)
                print('Time_Ground_mask = ', time.time()-time_init , "\n")

                time_init = time.time()
                stands_mask=getStandMask(groundColorMask, playerMask)
                cv2.imwrite('unique_stands_mask.png', stands_mask*255)
                # cv2.imwrite('stands_MASK.png', )
                print('Time_stand_mask = ', time.time()-time_init, "\n")
                time_init = time.time()

                rangeTH1,rangeTS1,rangeTV1,rangeTH2,rangeTS2,rangeTV2 = getPlayerColors(stands_mask, groundColorMask, img)
                print("ranges = ", [rangeTH1,rangeTH2])
                # st = np.zeros((1080,1920,3))
                # st[:,:,0]=st[:,:,1]=st[:,:,2]=stands_mask
                imgstar = cv2.imread('dmm_p.png')
                # all_players=cv2.imread('all_players.png',0)
                team1 = rangeToMask(imgstar,rangeTH1,rangeTS1,rangeTV1)
                team2 = rangeToMask(imgstar,rangeTH2,rangeTS2,rangeTV2)
                cv2.imwrite('team1.png', team1*255)
                cv2.imwrite('team2.png', team2*255)
                # kernel= np.ones((8,8), np.uint8)
                # team1=cv2.morphologyEx(team1,cv2.MORPH_CLOSE,kernel)
                # team1=1-team1
                # cv2.imwrite('team1_p.png', team1*255)
                upper_bound,lower_bound=findOuterBoundaries(stands_mask, playerMask)
                print([upper_bound,lower_bound])
                # print("files_jpg= ", files_jpg)
                player_boxes = getdata("SELECT player FROM tab WHERE frameID="+files_jpg[:-4], file_test_number)
                # print(len(player_boxes[0]))
                ball_boxes = getdata("SELECT ball FROM tab WHERE frameID="+files_jpg[:-4], file_test_number)
                # rangeTH1,rangeTS1,rangeTV1,rangeTH2,rangeTS2,rangeTV2 = getPlayerColors(groundColorMask,img)

                # print('player_boxes= ', player_boxes)


                print('Time_lb_ub_mask = ', time.time()-time_init, "\n")
                case, para1,para2,per1,per2 = find_inner_boundaries(upper_bound,lower_bound)
                lineMask1 = getLineMask(img,groundColorMask,playerMask,whiteL[0])
                sum1= np.sum(np.sum(lineMask1,axis=1),axis=0)
                err1 = error_in_linemask(lineMask1)
                # cv2.imwrite('lineMask1.png', lineMask1)

                lineMask2 = getLineMask(img,groundColorMask,playerMask,whiteR[0])
                sum2= np.sum(np.sum(lineMask2,axis=1),axis=0)
                err2 = error_in_linemask(lineMask2)
                # cv2.imwrite('lineMask2.png', lineMask2)

                lineMask3 = getLineMask(img,groundColorMask,playerMask,whiteC[0])
                sum3= np.sum(np.sum(lineMask3,axis=1),axis=0)
                err3 = error_in_linemask(lineMask3)
                # cv2.imwrite('lineMask3.png', lineMask3)

                if(err1 < err2 and err1 < err3):
                    lineMask=lineMask1
                    err = err1
                elif(err2 < err1 and err2 < err3):
                    lineMask = lineMask2
                    err = err2
                elif(err3 < err1 and err3 < err2):
                    lineMask = lineMask3
                    err = err3
                elif(err1==err2 and err1<err3):
                    if(sum1>sum2):
                        lineMask = lineMask1
                        err = err1
                    else:
                        lineMask = lineMask2
                        err = err2
                elif(err1==err2 and err1>err3):
                        lineMask = lineMask3
                        err = err3
                elif(err3==err2 and err1>err2):
                    if(sum3>sum2):
                        lineMask = lineMask3
                        err = err3
                    else:
                        lineMask = lineMask2
                        err = err2
                elif(err3==err2 and err1<err2):
                    lineMask = lineMask1
                    err = err1
                elif(err1==err3 and err1<err2):
                    if(sum1>sum3):
                        lineMask = lineMask1
                        err = err1
                    else:
                        lineMask = lineMask3
                        err = err3
                elif(err1==err3 and err1>err2):
                    lineMask = lineMask2
                    err = err2
                elif(err1 == err2 and err2==err3):
                    if(sum1>=sum2 and sum1>=sum3):
                        lineMask = lineMask1
                        err = err1
                    elif(sum2>=sum1 and sum2>=sum3):
                        lineMask = lineMask2
                        err = err2
                    else:
                        lineMask = lineMask3
                        err = err3

                # ques, ball = update_ball(img, lineMask, file_test_number,files_jpg[:-4])
                # print('query and ball = ',ques)
                # break
                buffer_queries.append([ques, ball])
                # print([buffer_queries])
                if(legit_frames(player_boxes[0], upper_bound, lower_bound)==1):
                    error, err_points = error_in_ground_mask(groundColorMask, playerMask)
                    error_threshold = 50
                    alpha_f+=1
                    print('alpha_f= ', alpha_f)
                    if(status==0):
                        if(error < error_threshold and err > 100):
                            ep = 1
                        elif(error>error_threshold and err < 100):
                            ep = 2
                        elif(error>error_threshold and err > 100):
                            ep = 3
                        else:
                            ep = 0
                    if(ep==2 or ep==3):
                    	# smay_i = time.time()
                        imgx = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        hsv_to_update = imgx[err_points]
                        hue = hsv_to_update[:,0]
                        sat = hsv_to_update[:,1]
                        valu= hsv_to_update[:,2]


                        cntsH = gethistogram(hue, 180)
                        leftH, rightH = getlr(cntsH, 20)
                        leftH = leftH/180
                        rightH = rightH/180
                        # print_histogram(cntsH)

                        cntsS = gethistogram(sat, 256)
                        leftS, rightS = getlr(cntsS, 20)
                        leftS = leftS/256
                        rightS = rightS/256
                        # print("prev range= ", rangeH)
                        rangeH = np.asarray([min(rangeH[0], leftH), max(rangeH[1], rightH)])
                        rangeS = np.asarray([min(rangeS[0], leftS), max(rangeS[1], rightS)])
                        # print("finalrange= ", rangeH)
                    if(ep==1 or ep==3):
                        thres = getLineRange(img,groundColorMask,playerMask)
                        if((case==1 or case==3 or case==4) and abs(thres-whiteL[0])<5):
                            whiteL = np.asarray([min(thres, whiteL[0]), 255])
                        elif((case==2 or case==5 or case==6) and abs(thres-whiteR[0])<5):
                            whiteR = np.asarray([min(thres, whiteR[0]), 255])
                        elif(abs(thres-whiteC[0])<5):
                            whiteC = np.asarray([min(thres, whiteC[0]), 255])

                        # smay_f = time.time() - smay_i
                        # print('smay_f= ',smay_f)

                    # if(status==0 or (status==2 and (ep==1 or ep==3))):
                    # 	conn = sqlite3.connect(DATABASE+str(file_test_number)+".db", detect_types=sqlite3.PARSE_DECLTYPES)
                    # 	c = conn.cursor()
                    # 	if(status==2):
                    # 		ep=1
                    # 	insert(iota+1, rangeH,rangeS,rangeV, whiteL, whiteC, whiteR,ep,file_name_jpg,case,c)
                    # 	# iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_ = getdata("SELECT * FROM color ORDER BY ID DESC LIMIT 1", file_test_number)
                    # 	# print([iota,rangeH, rangeS, rangeV, whiteL, whiteC, whiteR, status,_])
                    # 	conn.commit()
                    # 	print("hujsu")
                    # 	conn.close()

                else:
                    continue


                # if case == 1 or case == 3 or case == 4:
                # 	lineMask = getLineMask(img,groundColorMask,whiteL[0])
                # 	sum1= np.sum(np.sum(linemask,axis=1),axis=0)

                # elif case == 2 or case == 5 or case == 6:
                # 	lineMask = getLineMask(img,groundColorMask,whiteR[0])
                # else:
                # 	lineMask = getLineMask(img,groundColorMask,whiteC[0])

                # er=error_in_linemask(lineMask,s)
                # cv2.imwrite('lineMask'+str(s)+'.png', lineMask)
                # s+=1
                # err.append(er)

                coords = getEndPoints(lineMask)
                coords,lines = lineMerge(coords)

                if goalData != []:
                    bBox = goalData[0]
                    top1, top2, bottom1, bottom2 = getGoalLine(img, bBox, white_thres, view, *goalParams)
                    img = Draw_goal(img, top1, top2, bottom1, bottom2)


                img = Draw_Upper_Lower(img,groundColorMask,stands_mask,Sub_outerboundaries_save_Dir,files_jpg,upper_bound,lower_bound)
                Draw_cases_lines(Sub_outerboundaries_save_Dir,files_jpg,case,para2,per2,lineMask,coords)
                if(len(buffer_queries)>=20):
                    conn = sqlite3.connect(DATABASE+str(file_test_number)+".db", detect_types=sqlite3.PARSE_DECLTYPES)
                    c = conn.cursor()
                    # insert(iota+1, rangeH,rangeS,rangeV, whiteL, whiteC, whiteR,0,file_name_jpg,case,c)
                    for q in range(len(buffer_queries)):
                        c.execute(buffer_queries[q][0], (buffer_queries[1], ))
                    conn.commit()
                    conn.close()
                    buffer_queries=[]

    if(len(buffer_queries)>=20):
        conn = sqlite3.connect(DATABASE+str(file_test_number)+".db", detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        # insert(iota+1, rangeH,rangeS,rangeV, whiteL, whiteC, whiteR,0,file_name_jpg,case,c)
        for q in range(len(buffer_queries)):
            c.execute(buffer_queries[q][0], (buffer_queries[1], ))
        conn.commit()
        conn.close()
        buffer_queries=[]
    # for process in processes:
    # 	process.join()
            # except:
            # 	print("hallo")
