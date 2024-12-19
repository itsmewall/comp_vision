import cv2
import cvzone
from pyzbar import pyzbar as bar

cap = cv2.VideoCapture(0)

while 1:
    ret,frame = cap.read()

    result = bar.decode(frame)
    for data in result:
        print (data.data)  

    cv2.imshow('frame', frame)
    cv2.waitKey(1)