#headpose
#-----------------------------------------------------------------------------------
import cv2
import numpy as np
import math
import dlib
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
# import datetime
from datetime import datetime
import time
import cv2
from deepface import DeepFace
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
import csv
# Load face detector and landmark predictor
face_modeldlib = dlib.get_frontal_face_detector()
landmark_modeldlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def find_faces1(img, face_modeldlib):
    return face_modeldlib(img)

def detect_marks1(img, predictor, rect):
    shape = predictor(img, rect)
    return np.array([[p.x, p.y] for p in shape.parts()])

def draw_marks1(img, marks, color=(0, 255, 0)):
    for mark in marks:
        cv2.circle(img, tuple(mark), 2, color, -1)
font = cv2.FONT_HERSHEY_SIMPLEX


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float64).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways    

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)
    
face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

#mouth opening
#----------------------------------------------------------------------------
import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3

# Define mouth landmark points
outer_points = [(48, 54), (49, 53), (50, 52), (51, 57)]
inner_points = [(60, 64), (61, 63), (62, 66), (67, 65)]
d_outer = [5, 5, 5, 5]  # Thresholds
font = cv2.FONT_HERSHEY_SIMPLEX


#face spoofing
#----------------------------------------------------------------------------------
import numpy as np
import cv2
#from sklearn.externals import joblib
from face_detector import get_face_detector, find_faces
#import sklearn.external.joblib as extjoblib

import joblib
def calc_hist(img):
    """
    To calculate histogram of an RGB image

    Parameters
    ----------
    img : Array of uint8
        Image whose histogram is to be calculated

    Returns
    -------
    histogram : np.array
        The required histogram

    """
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

face_model = get_face_detector()
clf = joblib.load('models/face_spoofing.pkl')

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

#eye tracking
#----------------------------------------------------------------------
import cv2
import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img3, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img3, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img3, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
            timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            cv2.imwrite('dataset/looking left'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
            text = 'Looking left'
        elif left == 2:
            print('Looking right')
            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
            timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            cv2.imwrite('dataset/looking right'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
            text = 'Looking right'
        elif left == 3:
            print('Looking up')
            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
            timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            cv2.imwrite('dataset/looking up'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img3, text, (40, 40), font,  
                    1, (0, 255, 255), 2, cv2.LINE_AA) 

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

##    cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
##    cv2.createTrackbar('threshold', 'image', 75, 255, nothing)


#person and phone
#--------------------------------------------------------------
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["models/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["models/yolov3.weights"])
configPath = os.path.sep.join(["models/yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

WITDH = 450
HEIGHT = 350

f = open('emotions.csv', 'w', newline='')
f.close()

cap = cv2.VideoCapture(0)
def getFrame():
    ret, frame = cap.read()
    return frame

def HeadPose():
    while(True):
        try:
            frame = getFrame()
            img = cv2.resize(frame, (WITDH, HEIGHT))
            faces = find_faces(img, face_model)
            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],     # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                    ], dtype="double")
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                
                
                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose
                
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                
                
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                    
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                    
                    # print('div by zero error')
                if ang1 >= 48:
                    print('Head down')
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/head down'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    cv2.putText(img, 'Head down', (40, 40), font, 0.5, (255, 255, 128), 2)
                elif ang1 <= -48:
                    print('Head up')
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/head up'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    cv2.putText(img, 'Head up', (40, 40), font, 0.5, (255, 255, 128), 2)
                    
                if ang2 >= 48:
                    print('Head right')
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/head right'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    cv2.putText(img, 'Head right', (40, 40), font, 0.5, (255, 255, 128), 2)
                elif ang2 <= -48:
                    print('Head left')
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/head left'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    cv2.putText(img, 'Head left', (40, 40), font, 0.5, (255, 255, 128), 2)
                else:
                    print('Head center')
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/head center'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    cv2.putText(img, 'Head center', (40, 40), font, 0.5, (255, 255, 128), 2)
                
                cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 2)
                cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 2)
                
            cv2.putText(img, 'Head pose detection', (20, 20), font, 0.5, (255, 255, 128), 2)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('head error', e)

def MouthPose():
    while True:
        try:
            #mouth opening
            #-------------------------------------------------------------
            img1 = getFrame()
            # img1 = frame.copy()
            rects = find_faces1(img1, face_modeldlib)
            for rect in rects:
                shape = detect_marks1(img1, landmark_modeldlib, rect)
                cnt_outer, cnt_inner = 0, 0
                draw_marks1(img1, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        cnt_outer += 1 
                for i, (p1, p2) in enumerate(inner_points):
                    if d_outer[i] + 2 < shape[p2][1] - shape[p1][1]:
                        cnt_inner += 1
                
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour, Minute, Second = timeStamp.split(":")
                
                if cnt_outer > 2 and cnt_inner > 1:
                    status = 'Mouth open'
                else:
                    status = 'Mouth closed'
                
                cv2.imwrite(f'dataset/{status}_{date}_{Hour}_{Minute}_{Second}.jpg', img1)
                cv2.putText(img1, status, (40, 40), font, 0.5, (0, 255, 255), 2)
            
            cv2.putText(img1, 'Mouth detection', (20, 20), font, 0.5, (255, 255, 128), 2)
            # cv2.imshow("Mouth Detection", img1)
            ret, buffer = cv2.imencode('.jpg', img1)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Mouth error', e)

def facePose():
    count = 0
    while True:
        try:
            #face spoofing
            #----------------------------------------------------------------------------
            img2 = getFrame()
            faces = find_faces(img2, face_model)
            measures[count%sample_number]=0
            height, width = img2.shape[:2]
            for x, y, x1, y1 in faces:
                
                roi = img2[y:y1, x:x1]
                point = (0,0)
                
                img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

                ycrcb_hist = calc_hist(img_ycrcb)
                luv_hist = calc_hist(img_luv)

                feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
                feature_vector = feature_vector.reshape(1, len(feature_vector))

                prediction = clf.predict_proba(feature_vector)
                prob = prediction[0][1]

                measures[count % sample_number] = prob

                cv2.rectangle(img2, (x, y), (x1, y1), (255, 0, 0), 2)

                point = (x, y-5)

                # print (measures, np.mean(measures))
                if 0 not in measures:
                    text = "True"
                    ts=time.time()
                    date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                    timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour,Minute,Second=timeStamp.split(":")
                    cv2.imwrite('dataset/True'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                    if np.mean(measures) >= 0.7:
                        text = "False"
                        ts=time.time()
                        date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                        timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        Hour,Minute,Second=timeStamp.split(":")
                        cv2.imwrite('dataset/False'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img=img2, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                    thickness=2, lineType=cv2.LINE_AA)
                    else:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img=img2, text=text, org=point, fontFace=font, fontScale=0.9,
                                    color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
            count+=1
            cv2.putText(img2, 'Face spoofing detection', (20, 20), font, 0.5, (255, 255, 128), 2)
            ret, buffer = cv2.imencode('.jpg', img2)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Face error', e)

def eyePose():
    while True:
        try:
            #eye tracking
            #-------------------------------------------------------------------------------------------------------
            img3 = getFrame()
            rects = find_faces(img3, face_model)
            for rect in rects:
                shape = detect_marks(img3, landmark_model, rect)
                mask = np.zeros(img3.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                
                eyes = cv2.bitwise_and(img3, img3, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = 0.75
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)
                
                eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                print_eye_pos(img3, eyeball_pos_left, eyeball_pos_right)
                
            cv2.putText(img3, 'Eye tracking detection', (20, 20), font, 0.5, (255, 255, 128), 2)
            ret, buffer = cv2.imencode('.jpg', img3)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Eye error', e)

def Person():
    while True:
        try:
            #person and phone 
            #----------------------------------------------------------------
            frame = getFrame()
            img4 = frame.copy()

            (H, W) = frame.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                            # extract the class ID and confidence (i.e., probability)
                            # of the current object detection
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            confidence = scores[classID]

                            # filter out weak predictions by ensuring the detected
                            # probability is greater than the minimum probability
                            if confidence > args["confidence"]:
                                    # scale the bounding box coordinates back relative to
                                    # the size of the image, keeping in mind that YOLO
                                    # actually returns the center (x, y)-coordinates of
                                    # the bounding box followed by the boxes' width and
                                    # height
                                    box = detection[0:4] * np.array([W, H, W, H])
                                    (centerX, centerY, width, height) = box.astype("int")

                                    # use the center (x, y)-coordinates to derive the top
                                    # and and left corner of the bounding box
                                    x = int(centerX - (width / 2))
                                    y = int(centerY - (height / 2))

                                    # update our list of bounding box coordinates,
                                    # confidences, and class IDs
                                    boxes.append([x, y, int(width), int(height)])
                                    confidences.append(float(confidence))
                                    classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                    args["threshold"])

            # ensure at least one detection exists
            count=0
            if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])

                            obj = "{}".format(LABELS[classIDs[i]])
                            if obj == 'person':
                                count +=1
                                
                                # draw a bounding box rectangle and label on the frame
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(img4, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img4, obj, (x, y - 5),
                                        font, 0.5, color, 2)
                    f = open('personnum.txt', 'w')
                    f.write(str(count))
                    f.close()
                    if count>1:
                        ts=time.time()
                        date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                        timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        Hour,Minute,Second=timeStamp.split(":")
                        cv2.imwrite('dataset/person'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img4)
                                
            cv2.putText(img4, 'person detecton', (20, 20), font, 0.5, (255, 255, 128), 2)
            ret, buffer = cv2.imencode('.jpg', img4)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Person error', e)

def Phone():
    while True:
        try:
            #person and phone 
            #----------------------------------------------------------------
            frame = getFrame()
            img5 = frame.copy()
            (H, W) = frame.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                            # extract the class ID and confidence (i.e., probability)
                            # of the current object detection
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            confidence = scores[classID]

                            # filter out weak predictions by ensuring the detected
                            # probability is greater than the minimum probability
                            if confidence > args["confidence"]:
                                    # scale the bounding box coordinates back relative to
                                    # the size of the image, keeping in mind that YOLO
                                    # actually returns the center (x, y)-coordinates of
                                    # the bounding box followed by the boxes' width and
                                    # height
                                    box = detection[0:4] * np.array([W, H, W, H])
                                    (centerX, centerY, width, height) = box.astype("int")

                                    # use the center (x, y)-coordinates to derive the top
                                    # and and left corner of the bounding box
                                    x = int(centerX - (width / 2))
                                    y = int(centerY - (height / 2))

                                    # update our list of bounding box coordinates,
                                    # confidences, and class IDs
                                    boxes.append([x, y, int(width), int(height)])
                                    confidences.append(float(confidence))
                                    classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                    args["threshold"])

            # ensure at least one detection exists
            count=0
            if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])

                            obj = "{}".format(LABELS[classIDs[i]])
                            if obj == 'cell phone':
                                # draw a bounding box rectangle and label on the frame
                                ts=time.time()
                                date=datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                                timeStamp=datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                                Hour,Minute,Second=timeStamp.split(":")
                                cv2.imwrite('dataset/phone'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img5)
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(img5, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(img5, obj, (x, y - 5),
                                        font, 0.5, color, 2)

            cv2.putText(img5, 'Cell phone detection', (20, 20), font, 0.5, (255, 255, 128), 2)
            ret, buffer = cv2.imencode('.jpg', img5)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Phone error', e)

def Emotion():
    while True:
        try:
            img6 = getFrame()
            # cv2.imshow("emotion code",img6)
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                
                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emoname = result[0]['dominant_emotion']
                print(emoname)
                ["Frustration" ,"Disapproval","Nervousness", "Confident", "Disappointed", "Curiosity","Composure"]
                if emoname == "angry":
                    emotion="Frustration"
                elif emoname == "disgust":
                    emotion="Disapproval"
                elif emoname == "fear":
                    emotion="Nervousness"
                elif emoname == "happy":
                    emotion="Confident"
                elif emoname == "sad":
                    emotion="Disappointed"
                elif emoname == "surprise":
                    emotion="Curiosity"
                elif emoname == "neutral":
                    emotion="Composure"
                
                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(img6, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img6, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print(emotion)
                now = datetime.now() # current date and time
                date_time = now.strftime("%m%d%Y%H%M%S")
                cv2.imwrite('emotions/'+str(date_time)+'.png', img6)
                f = open('emotions.csv', 'a', newline='')
                writer = csv.writer(f)
                writer.writerow([emotion])
                f.close()
            ret, buffer = cv2.imencode('.jpg', img6)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print('Emotion error', e)