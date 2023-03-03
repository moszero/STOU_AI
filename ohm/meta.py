#import
import cv2
import mediapipe as mp
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import time
from datetime import datetime
import utils,math
import matplotlib.pyplot as plt
import pandas as pd
import os
import mss
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
import tqdm

mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     # เพิ่มจำนวน detect หน้าสูงสุด
#     max_num_faces=10,
# )

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
count = 0
seconds = time.time()
# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('video/source/2/CB_SOS_11042022.mp4')

ith_sample = 0
face_frame = 0
status = list('-'*20)

# variables 
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
mouthStatus = ''
# *-------------------------------*
#screen size
# monitor = {"top": 40, "left": 40, "width": 500, "height": 500}
try:
    if len(sys.argv) == 1:
        monitor = {"top": 100, "left": 0, "width": 1500, "height": 2000}
    else:
        mornitor_t = int(sys.argv[-1].split(',')[1])
        mornitor_l = int(sys.argv[-1].split(',')[0])
        mornitor_w = int(sys.argv[-1].split(',')[2])
        mornitor_h = int(sys.argv[-1].split(',')[3])
        monitor = {"top": mornitor_t, "left": mornitor_l, "width": mornitor_w, "height": mornitor_h}
except:
        monitor = {"top": 100, "left": 0, "width": 1500, "height": 1000}
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
ratio_mount = 0
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
#data list csv
#eye
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark
#mount
detector = FaceMeshDetector(maxFaces=2)
idList = [0,17,78,292]  

countF = 0
countL = 0
countR = 0
countU = 0
countD = 0
#eye
lookL = 0
lookR = 0
lookC = 0
blink = 0
#mount
mount = 0
X_test = []
import csv
header = ['h_forward','h_left', 'h_right', 'h_up', 'h_down','L_left', 'L_Right', 'L_center', 'blink', 'mount']
data_train = [
    [countF, countL,  countR, countU,  countD, lookL, lookR, lookC, blink, mount],
]
def build_CSV():
    with open('data_train_test2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        # Use writerows() not writerow()
        writer.writerows(data_train)
        
def landmarksDetection(img, landmark, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks 
        return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 


# Eyes Extrctor function,
#Funcao que define distancias euclidianas dos pontos nos olhos
def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

#Funcao para encontra posicao da iris
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio <= 0.42:
        iris_position="left"
        # cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,0,255),10)
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
        # cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(255,0,0),10)
    else:
        iris_position = "right"
        # cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,0,255),10)
    return iris_position, ratio

# landmark detection function 
with mp_face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    with mss.mss() as sct: #แคปหน้าจอ
        success = True
        print(mss.mss())
        while "Screen capturing":
            image = np.array(sct.grab(monitor))
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ith_sample += 1
            start = time.time()
            image, faces = detector.findFaceMesh(image,draw = False)
            
            image.flags.writeable = False

            # Get the result
            results = face_mesh.process(image)

            # To improve performance
            image.flags.writeable = True

            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            

            # ดูจำนวนคน
            # print('len(results.multi_face_landmarks)')
            cv2.putText(image, 'frame '+str(ith_sample), (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            #cv2.putText(image, ' '.join(status), (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            if results.multi_face_landmarks:
                face_frame += 1
                cv2.putText(image,'face frame '+ str(face_frame), (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)  
                # classIds, confs, bbox = net.detect(image,confThreshold=0.5)
                # for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                #     cv2.rectangle(image,box,color=(0,255,0),thickness=2)
                #     cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                    
                
                
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx,r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                # transforma pontos centrais em array np
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                #desenhe o círculo com base nos valores de retorno da minEnclosingCircle, através do CIRCLE que desenha a imagem do círculo com base no centro (x, y) e no raio
                cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
                #mostrar pontos nos cantos dos olhos
                cv2.circle(image, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
              
                # cv2.putText(image, f"ref : {iris_pos} {ratio:.2f}",(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)
                #eye//
                for i,face_landmarks in enumerate(results.multi_face_landmarks):
                    # เพิ่มกรอบหน้าขาว
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    
                    # ส่วนของการกระพริบตา
                    mesh_coords = landmarksDetection(image, face_landmarks.landmark, False)
                    try:
                        ratio = blinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    except:
                        print("blink error")
                    # ส่วนของการมอง
                    try:
                        iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
                    except:
                        print("iris_pos error")
                    # ส่วนของปาก
                    try:
                        if faces:
                            face = faces[0]
                            for id in idList:
                                cv2.circle(image,face[id],5,(255,0,255),5)
                            upDown,_ = detector.findDistance(face[idList[0]],face[idList[1]])
                            leftRight,_ = detector.findDistance(face[idList[2]],face[idList[3]])

                            ratio_mount = int((upDown/leftRight)*100)
                            if ratio_mount > 50:
                                mouthStatus = 'open'
                                cv2.line(image,face[idList[0]],face[idList[1]],(0,0,255),3)
                                cv2.line(image,face[idList[2]],face[idList[3]],(0,0,255),3)
                            else:
                                cv2.line(image,face[idList[0]],face[idList[1]],(0,255,0),3)
                                cv2.line(image,face[idList[2]],face[idList[3]],(0,255,0),3)
                                mouthStatus = 'close'
                    except:
                        print("mount err")
                    # cv2.putText(image, f'Ratio : {round(ratio,2)}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if ratio > 4.5: # ตรวจการกระพริบของตา
                        CEF_COUNTER +=1
                        # cv.putText(image, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                        utils.colorBackgroundText(image,  f'Blink', cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER>CLOSED_EYES_FRAME:
                            TOTAL_BLINKS +=1
                            CEF_COUNTER =0
                            
                    face_2d = []
                    face_3d = []
                    for idx, lm in enumerate(face_landmarks.landmark):

                        # เอาไว้หาตำแหน่งจุดของหน้า
                        # test_x, test_y = int(lm.x * img_w), int(lm.y * img_h)
                        # cv2.circle(image, (test_x, test_y), 2, (255, 0, 0), -1)
                        # cv2.putText(image, str(idx), (test_x, test_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # เก็บตำแหน่งบนหน้าผาก
                        if idx == 10:
                            header_x,header_y = int(lm.x * img_w), (int(lm.y * img_h))
                            color_nose = (255, 0, 0)
                        # เก็บตำแหน่งจมูก
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            # print('mosss',i)
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    start_point = (5, 220)
                    end_point = (440, 440)
                    color = (255, 0, 0)
                    thickness = 2
                    
                    #reset val
                    countF = 0
                    countL = 0
                    countR = 0
                    countU = 0
                    countD = 0
                    lookL = 0
                    lookR = 0
                    lookC = 0
                    blink = 0
                    mount = 0
                    # See where the user's head tilting
                    if(ratio_mount > 50):
                        mount = 1
                    if(ratio > 4.5):
                        blink = 1
                    if(iris_pos == 'right'):
                        lookR = 1
                    elif(iris_pos == 'left'):
                        lookL = 1
                    else:
                        lookC = 1
  
                    if angles[1] < -0.020:
                        text = "Looking Right"
                        count = count+1
                        countR = 1
                        if(iris_pos == 'right'):
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,0,255),10)
                        else:
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                        color_nose = (0, 0, 255)
                    elif angles[1] > 0.020:
                        text = "Looking Left"
                        count = count+1
                        countL = 1
                        if(iris_pos == 'left'):
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,0,255),10)
                        else:
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                        color_nose = (0, 0, 255)
                    elif angles[0] < -0.020:
                        text = "Looking Down"
                        countD = 1
                        cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                        color_nose = (0, 0, 255)
                    elif angles[0] > 0.020:
                        text = "Looking Up"
                        countU = 1
                        cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,255,0),10)
                        color_nose = (255, 0, 0)
                    else:
                        text = "Forward"
                        count = count-1
                        countF = 1
                        if(iris_pos == 'right'):
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                        elif(iris_pos == 'left'):
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                        else:
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,255,0),10)
                        color_nose = (255, 0, 0)
                    #get data    
                    data_train.append([countF, countL,  countR, countU,  countD, lookL, lookR, lookC, blink, mount])
                    if(len(results.multi_face_landmarks) >= 2):
                        cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0,0,255),10)
                    if(ratio > 4.5):
                        if(angles[0] < -0.025):
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 0, 255),10)
                        else:
                            cv2.rectangle(image,(0,0),(image.shape[1],image.shape[0]),(0, 165, 255),10)
                    

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                    cv2.line(image, p1, p2, color_nose, 3)
                    # Add the text on the image
                    cv2.putText(image, text, (header_x,header_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color_nose, 2)
                    cv2.putText(image, "x: " + str(np.round(angles[0], 2)), (header_x,header_y-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "y: " + str(np.round(angles[1], 2)), (header_x,header_y-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "z: " + str(np.round(z, 2)), (header_x,header_y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.putText(image,str(mouthStatus),(header_x,header_y-40),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)
                    cv2.putText(image, f"eye : {iris_pos} {ratio:.2f}",(header_x,header_y-20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)
                    cv2.putText(image, f"mouth : {mouthStatus} {ratio_mount:.2f}",(header_x,header_y-40),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)
                    # cv2.putText(image,str(ratio),(header_x,header_y+20),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),1)
                    # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    # cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime

                #cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Head Pose Estimation1', image)

            if cv2.waitKey(5) & 0xFF == 27:
                print('break')
                break
        cv2.destroyAllWindows()
        # cap.release()
