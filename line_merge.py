import math
import numpy as np

'''
Input: coords from end_points file.
Output: Combined coords
'''

def lineMerge(coords):
    '''
    lines contains the slope and intercept of corresponding line in coords in the form [m,c]
    '''

    lines=[]
    for point in coords:
      if(point[0][1]==point[1][1]):
          m=1e9
          c=point[0][0]-point[0][1]*m
      else:
          m = (point[1][0]-point[0][0])/(point[1][1]-point[0][1])
          c = point[0][0]-point[0][1]*m
      lines.append([m,c])
    # print("LINEs before : ",lines)
    # print("COORDS BEFORE:", coords)
    '''
    Merging of elements in lines and coords with almost same slopes and intercepts.
    Detection of center line index depending on slope (at least 80 degrees), minimum length (atleast 300) and choosing longest of all such lines.
    Center line index is -1 if no such line is detected.
    If center line is detected, removal of all lines which intesect center line at y>0 takes place.
    If center line is detected and lies comfortably in left/right part of the frame, then lines with positive/negative (actually negative/positive) slopes are removed.
    The points on the detected lines are plotted on them and stored as an image in a folder named 'ep' as 'z.png' where z is the input image name without extension.
    '''

    ep_m=15*(math.pi)/180
    ep_c = 20
    groupLabels = [0 for i in range(len(coords))]
    see_len=len(coords)
    # pairs_mer=[]
    for i in range(see_len):
      for j in range(i+1, see_len):
              x_s = [coords[i][0][1],coords[i][1][1],coords[j][0][1],coords[j][1][1]]
              x_s.sort()
              x_op = (x_s[1] + x_s[2])/2
              y_1= x_op * lines[i][0] + lines[i][1]
              y_2= x_op * lines[j][0] + lines[j][1]

              y_s = [coords[i][0][0],coords[i][1][0],coords[j][0][0],coords[j][1][0]]
              y_s.sort()
              y_op = (y_s[1] + y_s[2])/2
              x_1= (y_op - lines[i][1])/lines[i][0] if lines[i][0] != 0 else np.inf
              x_2= (y_op - lines[j][1])/lines[j][0] if lines[j][0] != 0 else np.inf

              m_1=lines[i][0]
              m_2=lines[j][0]
              if(abs(lines[i][0])>=5.67):
                  m_1 = abs(m_1)
              if(abs(lines[j][0])>=5.67):
                  m_2 = abs(m_2)
              if(abs(math.atan(m_1)-math.atan(m_2))<ep_m and groupLabels[i]==0 and groupLabels[j]==0):
                #   print(abs(x_1-x_2)<ep_c)
                  if(abs(m_1)>=5.67 and abs(m_2)>=5.67 and abs(x_1 - x_2)<ep_c):
                      li=[coords[i][0][0],coords[i][1][0],coords[j][0][0],coords[j][1][0]]
                      li_min=min(range(len(li)), key=li.__getitem__)
                      li_max=max(range(len(li)), key=li.__getitem__)
                      if(li_min//2 == 0):
                          in_min=i
                      else:
                          in_min=j
                      if(li_max//2 == 0):
                          in_max=i
                      else:
                          in_max=j
                      # print[i,j, in_min, in_max]
                      pi=[]
                      if(in_min==i):
                          pi.append(coords[i][li_min])
                      else:
                          pi.append(coords[j][li_min-2])
                      if(in_max==i):
                          pi.append(coords[i][li_max])
                      else:
                          pi.append(coords[j][li_max-2])
                      pnt= pi
                      coords[i]=pi
                      if(pnt[0][1]==pnt[1][1]):
                          m=1e9
                          c=pnt[0][0]-pnt[0][1]*m
                      else:
                          m = (pnt[1][0]-pnt[0][0])/(pnt[1][1]-pnt[0][1])
                          c = pnt[0][0]-pnt[0][1]*m
                      lines[i]=[m,c]
                      groupLabels[j]=1
                      # pairs_mer.append([i,j])
                  elif((abs(m_1)<5.67 or abs(m_2)<5.67) and abs(y_1 - y_2)<ep_c):
                      li=[coords[i][0][0],coords[i][1][0],coords[j][0][0],coords[j][1][0]]
                      li_min=min(range(len(li)), key=li.__getitem__)
                      li_max=max(range(len(li)), key=li.__getitem__)
                      if(li_min//2 == 0):
                          in_min=i
                      else:
                          in_min=j
                      if(li_max//2 == 0):
                          in_max=i
                      else:
                          in_max=j
                      # print[i,j, in_min, in_max]
                      pi=[]
                      if(in_min==i):
                          pi.append(coords[i][li_min])
                      else:
                          pi.append(coords[j][li_min-2])
                      if(in_max==i):
                          pi.append(coords[i][li_max])
                      else:
                          pi.append(coords[j][li_max-2])
                      pnt= pi
                      coords[i]=pi
                      if(pnt[0][1]==pnt[1][1]):
                          m=1e9
                          c=pnt[0][0]-pnt[0][1]*m
                      else:
                          m = (pnt[1][0]-pnt[0][0])/(pnt[1][1]-pnt[0][1])
                          c = pnt[0][0]-pnt[0][1]*m
                      lines[i]=[m,c]
                      groupLabels[j]=1
                      # pairs_mer.append([i,j])
    # print("Gp= ", groupLabels)
    # print("Pairs_merged= ",pairs_mer)
    coords1=[]
    lines1=[]
    for i in range(len(groupLabels)):
      if(groupLabels[i]==0):
          coords1.append(coords[i])
          lines1.append(lines[i])

    ###############

    center_line=[0,0]
    center_length=0
    center_line_idx=-1
    for i in range(len(lines1)):
      di1 = math.sqrt((coords1[i][0][1]-coords1[i][1][1])**2 + (coords1[i][0][0]-coords1[i][1][0])**2)
      if(abs(lines1[i][0])>=5.67 and center_length<di1 and di1>=300):
          center_length=di1
          center_line=lines1[i]
          center_coords=coords1[i]
          center_line_idx=i

    ###############

    # if center_line_idx!=-1:
    #   i = 0
    #   while i<len(lines1):
    #       if lines1[i]!=center_line:
    #           x_int = (lines1[i][1]-center_line[1])/(center_line[0]-lines1[i][0])
    #           y_int = lines1[i][0]*x_int + lines1[i][1]
    #           if y_int>0:
    #               lines1.pop(i)
    #               coords1.pop(i)
    #           else:
    #               i += 1
    #       else:
    #           i += 1

    #   ###############

    #   if (center_coords[0][1]+center_coords[1][1])/2 > 1310:
    #       i = 0
    #       while i<len(lines1):
    #           if lines1[i]!=center_line and (lines1[i][0]<=-0.6 or lines1[i][0]>=-0.3):
    #               lines1.pop(i)
    #               coords1.pop(i)
    #           else:
    #               i += 1

    #   if (center_coords[0][1]+center_coords[1][1])/2 < 510:
    #       i = 0
    #       while i<len(lines1):
    #           if lines1[i]!=center_line and (lines1[i][0]>=0.6 or lines1[i][0]<=0.3):
    #               lines1.pop(i)
    #               coords1.pop(i)
    #           else:
    #               i += 1

    coords = coords1
    lines = lines1
    # print("LINES : ",lines)
    # print("COORDS : ",coords)
    return np.array(coords), lines