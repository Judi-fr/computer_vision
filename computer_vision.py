import cv2
import mediapipe as mp
import numpy
import time

cap=cv2.VideoCapture(0)
mpSolution = mp.solutions.hands
hands = mpSolution.Hands()
mpDibujo = mp.solutions.drawing_utils

HoraAnt = 0
HoraAct = 0 

while True:
    sincronizado, img = cap.read()
    
    if not sincronizado:
        print("No se pudo establecer conexion con la camara.")
        break

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #cambia img a rgb xq mediapipe soporta solo rgb
    resultados = hands.process(imgRGB)

    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            mpDibujo.draw_landmarks(img, mano,mpSolution.HAND_CONNECTIONS)

    HoraAct = time.time()
    fps = 1 / (HoraAct-HoraAnt)
    HoraAnt = HoraAct

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)

    cv2.imshow("Manos",img)
    cv2.waitKey(1)


    