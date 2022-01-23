import cv2
import numpy as np
import face_recognition
import os

imagesPath = 'employees'
employeesImages = []
employeesEncodeImages = []
employeesNames = []
employeesList = os.listdir(imagesPath)

for employee in employeesList:
    curImg = cv2.imread(f"./{imagesPath}/{employee}")
    curImg = cv2.cvtColor(curImg , cv2.COLOR_BGR2RGB)
    employeesImages.append(curImg)
    employeesEncodeImages.append(face_recognition.face_encodings(curImg)[0])
    employeesNames.append(os.path.splitext(employee)[0])

print(employeesNames)

cap = cv2.VideoCapture(0)

while True:
    success,img  = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodedCurFrameFaces = face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeFace , faceLoc in zip(encodedCurFrameFaces , facesCurFrame) :
        matches = face_recognition.compare_faces(employeesEncodeImages , encodeFace)
        faceDis = face_recognition.face_distance(employeesEncodeImages , encodeFace)
        mostMatchIndex = np.argmin(faceDis)
        if(faceDis[mostMatchIndex] <= 0.52):
            name = employeesNames[mostMatchIndex]
            x1,y1,x2,y2 = faceLoc
            x1, y1, x2, y2 = x1*4,y1*4,x2*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)