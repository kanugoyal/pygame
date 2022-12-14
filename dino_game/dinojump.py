import cv2
import numpy as np
from time import sleep

i = 0
   
cap = cv2.VideoCapture(0)
directory = 'C:/Users/joaovitor/Desktop/Frames'

while True:
    sucess, img = cap.read()
    
    if sucess:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(gray, 155, 105)
        
        cv2.imwrite(f'{directory}/image_{i}.png', imgCanny)
        
        i += 1
        
        cv2.imshow('Canny', imgCanny)
        
        sleep(0.25)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break
        

#Release everything if job is finished
cap.release()
cv2.destroyAllWindows()