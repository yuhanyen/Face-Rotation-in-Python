import sys
import numpy as np
import cv2
import dlib
from math import atan2,degrees

def getAngle(x,y):
    #angle=np.arccos(x.dot(y)/(np.sqrt(x.dot(x))*np.sqrt(y.dot(y))))*360/2/np.pi    
    xDiff = y[0] - x[0]
    yDiff = y[1] - x[1]
    #print(degrees(atan2(yDiff, xDiff)))
    return (degrees(atan2(yDiff, xDiff)))

def getPoints(predictor, frame):
    points = []
    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor(predictor)
    #frame = cv2.imread(filename,1)
    #frame_resized = cv2.resize(frame, (300, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    for k,d in enumerate(detections):
        shape = predictor(clahe_image, d)
        for i in range(68):
            #cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
            points.append((int(shape.part(i).x), int(shape.part(i).y))) 
    #cv2.imshow("image", frame)
    #cv2.imwrite(filename[:-4] + "+Getpoint.jpg",frame)
    return points

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
            
    return points

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor = sys.argv[1]
filename = sys.argv[2]
frame = cv2.imread(filename)
rows,cols = frame.shape[:2]
#print(rows)
#print(cols)
points = getPoints(predictor, frame)
x = np.array(points[27])
y = np.array(points[30])
angle = getAngle(x,y)
#print(x)
#print(y)
print(angle)
M = cv2.getRotationMatrix2D((y[1],y[0]),angle-90,1)
dst = cv2.warpAffine(frame,M,(cols,rows))
points = getPoints(predictor, dst)
cv2.imwrite(filename[:-4] + "+Result.jpg",dst)
x = np.array(points[27])
y = np.array(points[30])
angle = getAngle(x,y)
#print(x)
#sprint(y)
print(angle)


face_points = points[0:17]
eyebrow_points_right = points[17:22]
eyebrow_points_left = points[22:27]
eye_right = points[36:42]
eye_left = points[42:48]
nose = points[27:36]
mouth_up_up = points[48:55]
mouth_up_down = points[60:65]
mouth_down_up = points[55:60]
mouth_down_down = points[65:68]