from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2 as cv

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    ear= (A+B)/(2.0*C)
    return ear


thresh=0.25
flag=0
frame_check=15
detect = dlib.get_frontal_face_detector() 
predict= dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
cap= cv.VideoCapture(0)

while True:
    ret, frame= cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    sub=detect(gray,0)
    for s in sub:
        shape=predict(gray,s)
        shape=face_utils.shape_to_np(shape)
        lefteye=shape[lStart:lEnd]
        righteye=shape[rStart:rEnd]
        leftear=eye_aspect_ratio(lefteye)
        rightear=eye_aspect_ratio(righteye)
        Ear=(leftear + rightear)/2.0
        lefteyehull=cv.convexHull(lefteye)
        righteyehull=cv.convexHull(righteye)
        cv.drawContours(frame,[lefteyehull],-1,(0,255,0),1)
        cv.drawContours(frame,[righteyehull],-1,(0,255,0),1)
        if Ear<thresh:
            flag=flag+1
            print(flag)
            if flag>=frame_check:
                cv.putText(frame,"***ALERT***",(10,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                mixer.music.play()
        else:
            flag=0    
    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cv.destroyAllWindows()
cap.release()    


