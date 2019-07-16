from imageOperatives import *

'''
Function to get the case value out of 14 cases
Trivial to understand.

RETURNS:
[req_case == Detected case value,

para_lines == Slope of detected para line,

para_lines1 == [First point x coordinate of detected para line, First point y coordinate of detected para line,
Second point x coordinate of detected para line, Second point y coordinate of detected para line],

per_lines == Slope of per line detected

per_lines1 == [First point x coordinate of detected per line, First point y coordinate of detected per line,
Second point x coordinate of detected per line, Second point y coordinate of detected per line]
]
'''

def find_inner_boundaries(upper_bound,lower_bound):
    case_in=0
    lower_bound=lower_bound.T
    flag=10
    line_yax=0
    line_xax=0
    para_lines1=np.array([])
    per_lines1=np.array([])
    para_lines=per_lines=None

    if (len(upper_bound)!=0):
        upper_bound=upper_bound.T
        a=upper_bound
        lins=a
        if (len(upper_bound[:,1]) == 3):
            dist1=math.sqrt((lins[0][0] - lins[1][0]) ** 2 + (lins[0][1] - lins[1][1]) ** 2)
            dist2=math.sqrt((lins[2][0] - lins[1][0]) ** 2 + (lins[2][1] - lins[1][1]) ** 2)
            a1=(lins[0][1] - lins[1][1]) / (lins[0][0] - lins[1][0])
            a2=(lins[1][1] - lins[2][1]) / (lins[1][0] - lins[2][0])
            lower_flg=0
            if (len(lower_bound)!=0):
                b=lower_bound
                blins=b
                if (len(lower_bound[:,1]) == 3):
                    lower_flg=1
                    if (abs(a1) - abs(a2) > 0):
                        #left
                        case_in=1
                        flag=0
                        para_lines=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                        per_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                        para_lines1=np.array([blins[0][0],blins[0][1],blins[1][0],blins[1][1]])
                        per_lines1=np.array([lins[0][0],lins[0][1],lins[1][0],lins[1][1]])
                    else:
                        #right
                        case_in=2
                        flag=0
                        para_lines=(blins[2][1] - blins[1][1]) / (blins[2][0] - blins[1][0])
                        per_lines=(lins[1][1] - lins[2][1]) / (lins[1][0] - lins[2][0])
                        para_lines1=np.array([blins[2][0],blins[2][1],blins[1][0],blins[1][1]])
                        per_lines1=np.array([lins[2][0],lins[2][1],lins[1][0],lins[1][1]])
            if (lower_flg != 1):
                flag=2
                if (abs(a1) - abs(a2) > 0):
                    if (dist1 > 350):
                        case_in=3
                        flag=0
                        para_lines=(lins[2][1] - lins[1][1]) / (lins[2][0] - lins[1][0])
                        per_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                        para_lines1=np.array([lins[2][0],lins[2][1],lins[1][0],lins[1][1]])
                        per_lines1=np.array([lins[0][0],lins[0][1],lins[1][0],lins[1][1]])
                    else:
                        case_in=4
                        para_lines=(lins[2][1] - lins[1][1]) / (lins[2][0] - lins[1][0])
                        para_lines1=np.array([lins[2][0],lins[2][1],lins[1][0],lins[1][1]])
                        flag=1
                else:
                    if (dist2 > 350):
                        case_in=5
                        flag=0
                        para_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                        per_lines=(lins[2][1] - lins[1][1]) / (lins[2][0] - lins[1][0])
                        para_lines1=np.array([lins[1][0],lins[1][1],lins[0][0],lins[0][1]])
                        per_lines1=np.array([lins[2][0],lins[2][1],lins[1][0],lins[1][1]])
                    else:
                        case_in=6
                        para_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                        para_lines1=np.array([lins[0][0],lins[0][1],lins[1][0],lins[1][1]])
                        flag=1
        if (len(upper_bound[:,1]) == 2):
            b=lower_bound
            blins=b
            if (len(lower_bound)!=0):
                if (len(lower_bound[:,1]) == 2 or len(lower_bound[:,1]) == 3):
                    c1=((2*1080) - blins[0][1] - blins[1][1]) / 2
                    c2=(lins[0][1] + lins[1][1]) / 2
                    if (len(lower_bound[:,1]) == 2):
                        if (c1 > c2):
                            if (c1 > 10):
                                para_lines=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                                para_lines1=np.array([blins[0][0],blins[0][1],blins[1][0],blins[1][1]])
                                flag=1
                                case_in=7
                            else:
                                flag=2
                                case_in=14
                        else:
                            if (c2> 10):
                                flag=1
                                case_in=10
                                para_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                                para_lines1=np.array([lins[0][0],lins[0][1],lins[1][0],lins[1][1]])
                            else:
                                flag=2
                                case_in=14
                    else:
                        flag=2
                        case_in=14
                    if (len(lower_bound[:,1]) == 3):
                        b1=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                        b2=(blins[1][1] - blins[2][1]) / (blins[1][0] - blins[2][0])
                        flag=1
                        if (abs(b1) - abs(b2) > 0):
                            case_in=8
                            para_lines=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                            para_lines1=np.array([blins[0][0],blins[0][1],blins[1][0],blins[1][1]])
                        else:
                            case_in=9
                            para_lines=(blins[1][1] - blins[2][1]) / (blins[1][0] - blins[2][0])
                            para_lines1=np.array([blins[2][0],blins[2][1],blins[1][0],blins[1][1]])
            else:
                flag=1
                case_in=10
                para_lines=(lins[1][1] - lins[0][1]) / (lins[1][0] - lins[0][0])
                para_lines1=np.array([lins[0][0],lins[0][1],lins[1][0],lins[1][1]])
    else:
        if (len(lower_bound)!=0):
            flag=1
            b=lower_bound
            blins=b
            if (len(lower_bound[:,1]) == 2):
                para_lines=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                para_lines1=np.array([blins[0][0],blins[0][1],blins[1][0],blins[1][1]])
                case_in=11
            if (len(lower_bound[:,1]) == 3):
                b1=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                b2=(blins[1][1] - blins[2][1]) / (blins[1][0] - blins[2][0])
                if (abs(b1) > abs(b2)):
                    case_in=12
                    para_lines=(blins[0][1] - blins[1][1]) / (blins[0][0] - blins[1][0])
                    para_lines1=np.array([blins[0][0],blins[0][1],blins[1][0],blins[1][1]])
                else:
                    case_in=13
                    para_lines=(blins[1][1] - blins[2][1]) / (blins[1][0] - blins[2][0])
                    para_lines1=np.array([blins[2][0],blins[2][1],blins[1][0],blins[1][1]])
        else:
            flag=2
            case_in=14

    return [case_in, para_lines, para_lines1, per_lines, per_lines1]