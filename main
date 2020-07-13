import cv2                              
import numpy as np
import math

#   #creating camera object
cap = cv2.VideoCapture(0)                 

while( cap.isOpened() ) :
    kernel = np.ones((3,3),np.uint8)
    
    ret,img = cap.read()

#   #mirrored image
    img = cv2.flip(img,1)

#   #filter for decreasing noise
    img = cv2.bilateralFilter(img,10,50,100)

#   #convert to HSV 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         
#   # define range of skin color in HSV ()
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    
#   #extract skin colur image 
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
#   #extrapolate the hand to fill dark spots within and gauss approx 
    blur = cv2.dilate(mask,kernel,iterations = 4)    
    blur = cv2.GaussianBlur(blur,(11,11),0)    

#   #setting up threshold for binary colors
    ret,th1 = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#   #avoiding crash when no skin color found on the input
    try:
#       #finding contours

        contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        drawing = np.zeros(img.shape,np.uint8)
        max_area=0
       
        for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
        cnt=contours[ci]
        hull = cv2.convexHull(cnt)

        cv2.drawContours(drawing,[cnt],0,(0,255,128),2) 
        cv2.drawContours(drawing,[hull],0,(0,0,255),2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = contours[ci]     
         
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                cnt = 0
                o_cnt = 0

#               # calculate the angle
                for i in range(defects.shape[0]): 
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    
                   
                    
                    if angle > math.pi / 2 and angle <= math.pi*(2/3) :  # angle greater than 90 degree
                        o_cnt += 1
                        cv2.circle(drawing, far, 6, [0, 255, 255], -1)
                    
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                    
                    
        
                        
                if cnt == 1:
                    cv2.putText(img,'peace',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif cnt == 2 and o_cnt < 1:
                    cv2.putText(img,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)                    
                elif 1 <= o_cnt < 3 and cnt == 2:
                    cv2.putText(img,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(img,'',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

#    #no skin on  the input
    except:
        cv2.putText(img,'No skin found',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        
#   #showing windows
    cv2.imshow('input',img)   
    cv2.imshow('drawing',drawing)

#   #waiting for pressing "ESC"
    k = cv2.waitKey(10)
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
