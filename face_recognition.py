import cv2
import numpy as np

fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id = raw_input('enter user id')
num = 701
while(True):
    ret,img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(img,1.3,5);

    for(x,y,w,h) in faces:
       num = num + 1
       cv2.imwrite("face_recognition_dataset/"+str(id)+"."+str(num)+".jpg",img[y:y+h,x:x+w])
       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
       cv2.waitKey(10)
    cv2.imshow("Face",img);
    cv2.waitKey(1)
    if (num>800):
       break;

cam.release()
cv2.destroyAllWindows()
