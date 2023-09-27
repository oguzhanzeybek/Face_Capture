import numpy as np
import cv2

vid = cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
left_eye=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
upper_body=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
lower_body=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lowerbody.xml")




while(True):
    ret,frame= vid.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces=face_cascade.detectMultiScale(gray, 1.5,5)
    for(x,y,z,h) in faces:
       cv2.rectangle(frame,(x,y),(x + z ,y + h),[15, 255, 12],2)

 
#   eye=eye_cascade.detectMultiScale(gray,1.5,5)
#   for(x,y,z,h) in eye:
#       cv2.rectangle(frame,(x,y),(x + z ,y + h),[15, 255, 12],2)

#    eye_left=left_eye.detectMultiScale(gray,1.3,5)
#   for(x,y,z,h) in eye_left:
#        cv2.rectangle(frame,(x,y),(x + z ,y + h),[255, 0, 0],2) 

    eye_right=right_eye.detectMultiScale(gray,1.3,5)
    for(x,y,z,h) in eye_right:
        cv2.rectangle(frame,(x,y),(x + z ,y + h),[255, 255, 0],2)

#    smile=smile_cascade.detectMultiScale(gray2,1.5,5)
 #   for(x,y,z,h) in smile:
  #      cv2.rectangle(frame,(x,y),(x + z ,y + h),[255, 255, 0],2)    

  
    body_up=upper_body.detectMultiScale(gray,1.5,5)
    for(x,y,z,h) in body_up:
        cv2.rectangle(frame,(x,y),(x + z ,y + h),[255, 255, 0],2)


    body_low=lower_body.detectMultiScale(gray,1.5,5)
    for(x,y,z,h) in body_low:
        cv2.rectangle(frame,(x,y),(x + z ,y + h),[255, 255, 0],2)
    





    cv2.imshow("Oguzhan Zeybek",frame)

    
    
    if cv2.waitKey(1) & 0xFF == ord("c"):
        break

vid.release()
cv2.destroyAllWindows()    