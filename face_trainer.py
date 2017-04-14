import os
import cv2
import numpy as np
from PIL import Image

rec = cv2.createLBPHFaceRecognizer()
path = 'face_recognition_dataset'

def result(path):
    imagepath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagepath)
    faces=[]
    ids=[]
    for image_path in imagepath:
        face = Image.open(image_path)
        face_np =  np.array(face,'uint8')
        id =int(os.path.split(image_path)[-1].split('.')[0])
        faces.append(face_np)
        ids.append(id)
        cv2.imshow("training",face_np)
        cv2.waitKey(1)
        print id
    return np.array(ids),faces
   
ids,faces = result(path)
rec.train(faces,np.array(ids))
rec.save('face_training/trainingData.yml')
cv2.destroyAllWindows()

