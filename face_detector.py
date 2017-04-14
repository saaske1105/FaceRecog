import cv2
import numpy as np

fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.createLBPHFaceRecognizer()
rec.load("face_training/trainingData.yml")
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)
name= ["Pratik","Sangeet","Rishik","Vikku","Ausaf","Faf","Prit","Punit"]
while(True):
    ret,img = cam.read()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(img,1.3,5);

    for(x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
       id,conf = rec.predict(img[y:y+h,x:x+w])
       cv2.cv.PutText(cv2.cv.fromarray(img),name[id-1],(x,y+h),font,255)
       
    cv2.imshow("Face",img);
    k=cv2.waitKey(1)
    if (k==27):
       break;

cam.release()
cv2.destroyAllWindows()
