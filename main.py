import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )


    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyesCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(5, 5)

        )
        if len(eyes) >= 2:
           print(len(eyes),": eyes")
        for (ex, ey, ew, eh) in eyes:


            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            roi_gray = gray[ey:ey + eh, ex:ex + ew]
            roi_color = img[ey:ey + eh, ex:ex + ew]
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()