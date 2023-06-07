#Read Image
from sre_constants import SUCCESS
import cv2
import numpy as np
img = cv2.imread('Data\dog.png')
kernel = np.ones((5,5), np.uint8)

#Show Image
cv2.imshow("DOG",img)


#Gray Image
imGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",imGray)


#Blur Image
imBlur = cv2.GaussianBlur(imGray, (7,7), 0)
cv2.imshow("Blur Image",imBlur)


#Canny Image
imCanny = cv2.Canny(img, 100, 100)
cv2.imshow("Canny Image",imCanny)


#Dilation Image
imDailation  = cv2.dilate(imCanny,kernel, iterations=1)
cv2.imshow("Dailation Image",imDailation)


#Erosion Image
imErode = cv2.erode(imDailation, kernel, iterations=1)
cv2.imshow("Erosion Image",imErode)


#Resize Image(Smaller than orignal)
imResize = cv2.resize(img, (150,200))
cv2.imshow("resize Image",imResize)


#Cropped Image
imCropped = img[30:200,100:400]
cv2.imshow("Cropped Image",imCropped)


#Join Images
imhor = np.hstack((img,img,img))
cv2.imshow("Combine Image",imhor)


#Insertion of shapes
cv2.rectangle(img,(0,0),(100,150),(0,0,255),3)
cv2.circle(img,(200,50),30,(255,0,0),3)
cv2.imshow("image with shapes",img)


#Read and save the vedio
frameWidth = 640
frameHeight = 460
cap = cv2.VideoCapture('Data\sample-1.mp4')

while True:
    SUCCESS, img2 = cap.read()
    img = cv2.resize(img2,(frameWidth, frameHeight)) 
    cv2.imshow("vedio", img2)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break


#Face detection for group image
imGroup = cv2.imread('Data\Group.jpeg.jpg')
imGray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier('F:\internship assingment 2\haarcascade_frontalface_alt.xml')
faces = faceCascade.detectMultiScale(imGray1, 1,1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),3)
    cv2.imshow("result", img)
    cv2.waitKey(0)
