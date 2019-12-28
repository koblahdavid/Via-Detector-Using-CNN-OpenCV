# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 02:20:23 2018

@author: David
"""

#import pandas as pd
import numpy as np
import scipy.io as sio
from PIL import Image
#import matplotlib.pyplot as plt
import cv2
import sys
import timeit
#from sklearn.cluster import KMeans
start = timeit.default_timer()

matz = sio.loadmat(r'C:\Users\David\Desktop\SURF\Data\PCBData\filteredAlignedData.mat')
data_z=matz['filtBoardStack']



#######################################
iu = 0
ou = 0
au = 0
size = 0
cent_x = 0
cent_y = 0
wye = 0
ex = 0
actual_0 = 0
actual_1 = 0
actual_2 = 0
total = [[0 for i in range(3*1)]for j in range (1*1)]
trainee = [np.array([0 for i in range(1*3)]).T for j in range (405)]

for u in range (0, 405):
 trainee[u]= np.reshape(trainee[u],(1,3))

def slicer(photo):
    global au
    global cent_x
    global cent_y
    global size
    global actual_0
    global actual_1
    global actual_2
    global ex
    global wye
    global total
    global progeny

    
	# Capture frame-by-frame
   # frame = cv2.imread(photo) 
    frame = np.reshape(photo,(26,26))
    #frame = np.uint8(photo)
    

    gray  =np.array(Image.fromarray(photo), np.uint8)
    #frame = np.asarray(frame)
    	# load the image, clone it for output, and then convert it to grayscale
    			
    output = frame.copy()
#    
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.medianBlur(gray,5)
#    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#    gray = cv2.filter2D(gray, -1, kernel)

    gray	=	cv2.fastNlMeansDenoising(gray)
   # gray	=	cv2.fastNlMeansDenoising(gray)




#    gray = cv2.Canny(gray, 75, 200)
  
#    cv2.imwrite("Open_CV\Gray%d.png"% au, gray)	
    	# detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,10 ,
                            param1= 200, #20, 150
                            param2= 8, #10, 13
                            minRadius=2, #5
                            maxRadius=7)  #15
    	# print circles
    	
    	# ensure at least some circles were found
        
    if circles is None:
        pass
    
    else:
        circles = np.uint16(np.around(circles))
        cent_x = circles
        actual_0 = cent_x[:,:,0] + (ex)
        actual_0 = np.transpose(actual_0)
        actual_1 = cent_x[:,:,1] + (wye)
        actual_1 = np.transpose(actual_1)
        actual_2 = cent_x[:,:,2]
        actual_2 = np.transpose(actual_2)
        progeny = np.hstack((actual_0, actual_1, actual_2))
        total = np.vstack((total, progeny))
        if au == 0:
            cent_y = cent_x
        else:
            cent_y = np.concatenate((cent_y,cent_x), axis =1 )
        
      
        
        for i in circles[0,:]:
             
        # draw the outer circle
             cv2.circle(output,(i[0],i[1]),i[2],(0,0,255),1)
         #draw the center of the circle
           #  cv2.circle(output,(i[0],i[1]),1,(0,0,255),3)
  
        	# Display the resulting frame
        
        #cv2.imwrite("Open_CV\X_Via_%d_%d_%d.png"%(au, wye, ex), output)
        au = au + 1
        
############################################################################        
        
def smicer(photo):
    global ou
    global cent_x
    global cent_y
    global size
    global actual_0
    global actual_1
    global actual_2
    global ex
    global wye
    global total
    global progeny
	# Capture frame-by-frame

   # frame = cv2.imread(photo) 
    frame = np.reshape(photo,(26,26))
    #frame = np.uint8(photo)

    gray  =np.array(Image.fromarray(photo), np.uint8)
    #frame = np.asarray(frame)
    	# load the image, clone it for output, and then convert it to grayscale
    			
    output = frame.copy()
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.medianBlur(gray,5)
#    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#    gray = cv2.filter2D(gray, -1, kernel)
    gray	=	cv2.fastNlMeansDenoising(gray)
   # gray	=	cv2.fastNlMeansDenoising(gray)


    
    	# detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,10,
                            param1=200, #20, 150
                            param2=8, #10, 13
                            minRadius=8, #5
                            maxRadius=11)  #15
    	# print circles
    	
    	# ensure at least some circles were found
        
    if circles is None:
        pass
    
    else:
        circles = np.uint16(np.around(circles))
        cent_x = circles
        actual_0 = cent_x[:,:,0] + (ex)
        actual_0 = np.transpose(actual_0)
        actual_1 = cent_x[:,:,1] + (wye)
        actual_1 = np.transpose(actual_1)
        actual_2 = cent_x[:,:,2]
        actual_2 = np.transpose(actual_2)
        progeny = np.hstack((actual_0, actual_1, actual_2))
        total = np.vstack((total, progeny))
       
        
        if ou == 0:
            cent_y = cent_x
        else:
            cent_y = np.concatenate((cent_y,cent_x), axis =1 )
        
      
        
        for i in circles[0,:]:
             
        # draw the outer circle
             cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),1)
         #draw the center of the circle
 #            cv2.circle(output,(i[0],i[1]),1,(0,0,255),3)
  
        	# Display the resulting frame
            
         
       # cv2.imwrite("Open_CV\Y_Via_%d_%d_%d.png"%(ou, wye, ex), output)
        ou = ou + 1
 #############################################################################       
  
def sticer(photo):
    global iu
    global cent_x
    global cent_y
    global size
    global actual_0
    global actual_1
    global actual_2
    global ex
    global wye
    global total
    global progeny
	# Capture frame-by-frame

   # frame = cv2.imread(photo) 
    frame = np.reshape(photo,(26,26))
    #frame = np.uint8(photo)

    gray  =np.array(Image.fromarray(photo), np.uint8)
    #frame = np.asarray(frame)
    	# load the image, clone it for output, and then convert it to grayscale
    			
    output = frame.copy()
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.medianBlur(gray,5)
#    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#    gray = cv2.filter2D(gray, -1, kernel)
#    
    
    gray	=	cv2.fastNlMeansDenoising(gray)
    #gray	=	cv2.fastNlMeansDenoising(gray)


#
#    gray = cv2.Canny(gray, 75, 200)
  
    
    	# detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,5,
                            param1=250, #20, 150
                            param2=10, #10, 13
                            minRadius=12, #5
                            maxRadius=18)  #15
    	# print circles
    	
    	# ensure at least some circles were found
        
    if circles is None:
        pass
    
    else:
        circles = np.uint16(np.around(circles))
        cent_x = circles
        actual_0 = cent_x[:,:,0] + (ex)
        actual_0 = np.transpose(actual_0)
        actual_1 = cent_x[:,:,1] + (wye)
        actual_1 = np.transpose(actual_1)
        actual_2 = cent_x[:,:,2]
        actual_2 = np.transpose(actual_2)
        progeny = np.hstack((actual_0, actual_1, actual_2))
        total = np.vstack((total, progeny))
        
        
        if iu == 0:
            cent_y = cent_x
        else:
            cent_y = np.concatenate((cent_y,cent_x), axis =1 )
        
      
        
        for i in circles[0,:]:
             
        # draw the outer circle
             cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),1)
         #draw the center of the circle
             #cv2.circle(output,(i[0],i[1]),1,(0,0,255),3)
         
        #cv2.imwrite("Open_CV\Z_Via_%d_%d_%d.png"%(iu, wye, ex), output)
        iu = iu + 1   

##########################################################################



    
def Networker(fa):
    images = np.zeros((20164,26,26))
    main_counter = 0 
    global ex
    global wye
    global total
    global trainee
    global df
    
    for y in range(0, 142):
            for x in range(0, 142):
                pos_x = (13*x)
                prev_x = pos_x + 26
                if y==141 :
                    pos_y = 13*(y-1)
                    prev_y = pos_y + 26           
                else :
                    pos_y = (13*y)
                    prev_y = pos_y + 26
                    
                    
                images[main_counter] = data_z[pos_y:prev_y, pos_x:prev_x, fa]
                main_counter = main_counter + 1
    
    
    from keras.models import load_model
    labeler = load_model('Model_95_63.h5')
    
    X_tee = np.array(images).reshape(20164,26,26,1)
    
    y_pred = labeler.predict(X_tee)
      
            
            
    i = 0
    for y in range(0, 142):
        for x in range(0, 142):
            i= i+1
            if (y_pred[i-1] > 0.16):
                wye = y*13
                ex = x*13
                #cv2.imwrite("Img.png", images[i-1])
                images[i-1] = np.uint8(images[i-1])
                slicer(images[i-1])
                smicer(images[i-1])
                sticer(images[i-1])
#                df.to_csv('CNN_Results\Coordinates_Net_%d.csv'%(fa+1), index= False)
                
    trainee[fa]= np.vstack((trainee[fa], total))
        

#    df.to_csv('CNN_Results\Coordinates_Net_%d.csv'%(fa+1), index= False)
    temp = data_z[0:1846, 0:1846, fa]
    cv2.imwrite('CNN_Results\Slice_%d.png'%(fa+1),temp)        

##################################################################
def marker(num):

    global trainee
    
    chosen = trainee[num]

    new_data = [[0 for i in range(3*1)]for j in range (1*1)]
    
    for h in range (0, len(chosen)):
        grade = 0
        for g in range (215, 255):
            for q in range(0, len(trainee[g])):
                if abs(chosen[h,0]-trainee[g][q][0])<4 and abs(chosen[h,1]-trainee[g][q][1])<4 and abs(chosen[h,2]-trainee[g][q][2])<4:
                    grade=grade+1
                else:
                    pass
        if grade > 20 :
            fi = [[chosen[h,0], chosen[h,1], chosen[h,2]] ]
            new_data = np.concatenate([new_data, fi], axis = 0)
            grade = 0
        else:
            pass  
    

    uniqe = np.unique(new_data, axis = 0)
    
    
    
    
    for j in range(0, len(uniqe)):
       if (0<uniqe[j,2]<=7 ):
           uniqe[j,2] =  5
       elif (7<uniqe[j,2]<=11 ):
           uniqe[j,2] =  8
       elif (11<uniqe[j,2]<20 ):
           uniqe[j,2] =  12
    

    for q in range (0, len(uniqe)):
        for r in range (0, len(uniqe)):
            if uniqe[q,2]<5:
                if (uniqe[q,0]-9 <uniqe[r,0]<uniqe[q,0]+9) and (uniqe[q,1]-9 <uniqe[r,1]<uniqe[q,1]+9):
                    if (uniqe[r,2]<=uniqe[q,2]):
                        uniqe[r,0] = uniqe[q,0]
                        uniqe[r,1] = uniqe[q,1]
                        uniqe[r,2] = uniqe[q,2]
                    else:
                        pass
                else:
                    pass  
            elif 6<uniqe[q,2]<11:
                if (uniqe[q,0]-16 <uniqe[r,0]<uniqe[q,0]+16) and (uniqe[q,1]-16 <uniqe[r,1]<uniqe[q,1]+16) :
                    if (uniqe[r,2]<=uniqe[q,2]):
                        uniqe[r,0] = uniqe[q,0]
                        uniqe[r,1] = uniqe[q,1]
                        uniqe[r,2] = uniqe[q,2]
                    else:
                        pass
                else:
                    pass   
            elif 10<uniqe[q,2]<20:
                if (uniqe[q,0]-25 <uniqe[r,0]<uniqe[q,0]+25) and (uniqe[q,1]-25 <uniqe[r,1]<uniqe[q,1]+25 ):
                    if (uniqe[r,2]<=uniqe[q,2]):
                        uniqe[r,0] = uniqe[q,0]
                        uniqe[r,1] = uniqe[q,1]
                        uniqe[r,2] = uniqe[q,2]
                    else:
                        pass
                else:
                    pass    
            
    uniqe = np.unique(uniqe, axis = 0)
    

           
    x = uniqe[:, 0]
    y = uniqe[:, 1]
    r = uniqe[:, 2]
   
    board = cv2.imread('CNN_Results\Slice_%d.png'%num)
    for i in range(0, len(uniqe)):
        if 0<r[i]<6:
            cv2.circle(board,(x[i],y[i]),r[i],(0,255,0),4)
        elif 5<r[i]<10:
            cv2.circle(board,(x[i],y[i]),r[i],(0,255,0),4)
        elif r[i]==0:
            pass
        else:
            cv2.circle(board,(x[i],y[i]),r[i],(0,255,0),4)
        
    cv2.imwrite("Refined\Board_%d.png"%num, board)  
################################################################
    
for ta in range (215,255):
    Networker(ta)  
    iu = 0
    ou = 0
    au = 0
    size = 0
    cent_x = 0
    cent_y = 0
    wye = 0
    ex = 0
    actual_0 = 0
    actual_1 = 0
    actual_2 = 0
    total = [[0 for i in range(3*1)]for j in range (1*1)]
    progeny = 0

for x in range (230, 240):
    marker(x)
    
stop = timeit.default_timer()
total_time = stop - start

mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
