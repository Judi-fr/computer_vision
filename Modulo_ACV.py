import cv2
import mediapipe as mp
import numpy
import time


class handDetector():

    def __init__(self,mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpSolution = mp.solutions.hands
        self.hands = self.mpSolution.Hands(self.mode, self.maxHands,self.model_complexity,
                                            self.detectionCon, self.trackCon)
        self.mpDibujo = mp.solutions.drawing_utils


    def findHands(self, img):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #cambia img a rgb xq mediapipe soporta solo rgb
        resultados = self.hands.process(imgRGB)
        
        if resultados.multi_hand_landmarks:
            for mano in resultados.multi_hand_landmarks:
                self.mpDibujo.draw_landmarks(img, mano,self.mpSolution.HAND_CONNECTIONS)
        
        return img


def main():

    cap=cv2.VideoCapture(0)

    HoraAnt = 0
    HoraAct = 0 

    detector = handDetector()

    while True:
        
        sincronizado, img = cap.read()
    
        img = detector.findHands(img)

        if not sincronizado:
            print("No se pudo establecer conexion con la camara.")
            break
    
        HoraAct = time.time()
        fps = 1 / (HoraAct-HoraAnt)
        HoraAnt = HoraAct

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                    (255,0,255),3)

        cv2.imshow("Manos",img)
        cv2.waitKey(1)



main()