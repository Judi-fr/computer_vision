import cv2
import mediapipe as mp
import numpy
import time

cap=cv2.VideoCapture(0)
mpSolution = mp.solutions.hands
hands = mpSolution.Hands()


while True:
    sincronizado, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #cambia img a rgb xq mediapipe soporta solo rgb
    
                                                 
    if not sincronizado:
        print("No se pudo establecer conexion con la camara.")
        break
    
    cv2.imshow("Manos",img)
    cv2.waitKey(1)
    