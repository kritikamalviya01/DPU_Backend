# ### General imports ###
# from __future__ import division
# import numpy as np
# import pandas as pd
# import time
# from time import sleep
# import re
# import os
# import requests
# import argparse
# from collections import OrderedDict

# ### Image processing ###
# import cv2
# from scipy.ndimage import zoom
# from scipy.spatial import distance
# import imutils
# from scipy import ndimage
# import dlib
# from imutils import face_utils

# ### Model ###
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K

# def facial_emotion(filename):
#     """
#     Video streaming generator function.
#     """

#     # Start video capute. 0 = Webcam, 1 = Video file, -1 = Webcam for Web
#     video_capture = cv2.VideoCapture(os.path.join('recorded_video',filename))

#     # Image shape
#     shape_x = 48
#     shape_y = 48
#     input_shape = (shape_x, shape_y, 1)

#     # We have 7 emotions
#     nClasses = 7

#     # Timer until the end of the recording
#     end = 0

#     # Load the pre-trained X-Ception model
#     model = load_model('Models/video.h5')

#     # Load the face detector
#     face_detect = dlib.get_frontal_face_detector()

#     # Load the facial landmarks predictor
#     predictor_landmarks  = dlib.shape_predictor("Models/face_landmarks.dat")

#     # Prediction vector
#     predictions = []

#     # Timer
#     global k
#     k = 0
#     max_time = 15
#     start = time.time()

#     angry_0 = []
#     disgust_1 = []
#     fear_2 = []
#     happy_3 = []
#     sad_4 = []
#     surprise_5 = []
#     neutral_6 = []

#     # Record for 45 seconds
#     while video_capture.isOpened():
        
#         k = k+1
#         end = time.time()
        
#         # Capture frame-by-frame the video_capture initiated above
#         ret, frame = video_capture.read()
        
#         # Face index, face by face
#         face_index = 0
        
#         # Image to gray scale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # All faces detected
#         rects = face_detect(gray, 1)
        
#         #gray, detected_faces, coord = detect_face(frame)
        
        
#         # For each detected face
#         for (i, rect) in enumerate(rects):
            
#             # Identify face coordinates
#             (x, y, w, h) = face_utils.rect_to_bb(rect)
#             face = gray[y:y+h,x:x+w]
            
#             # Identify landmarks and cast to numpy
#             shape = predictor_landmarks(gray, rect)
#             shape = face_utils.shape_to_np(shape)
            
#             # Zoom on extracted face
#             face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            
#             # Cast type float
#             face = face.astype(np.float32)
            
#             # Scale the face
#             face /= float(face.max())
#             face = np.reshape(face.flatten(), (1, 48, 48, 1))
            
#             # Make Emotion prediction on the face, outputs probabilities
#             prediction = model.predict(face)
            
#             # For plotting purposes with Altair
#             angry_0.append(prediction[0][0].astype(float))
#             disgust_1.append(prediction[0][1].astype(float))
#             fear_2.append(prediction[0][2].astype(float))
#             happy_3.append(prediction[0][3].astype(float))
#             sad_4.append(prediction[0][4].astype(float))
#             surprise_5.append(prediction[0][5].astype(float))
#             neutral_6.append(prediction[0][6].astype(float))
            
#             # Most likely emotion
#             prediction_result = np.argmax(prediction)
            
#             # Append the emotion to the final list
#             predictions.append((prediction_result))

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  
#             break
#         # For flask, save image as t.jpg (rewritten at each step)
#         cv2.imwrite('tmp/t.jpg', frame)
        
#         # Emotion mapping
#         #emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
        
#         # Once reaching the end, write the results to the personal file and to the overall file
#         if end-start > max_time - 1 :
#             with open("static/js/db/histo_perso.txt", "w") as d:
#                 d.write("density"+'\n')
#                 for val in predictions :
#                     d.write(str(val)+'\n')
                
#             with open("static/js/db/histo.txt", "a") as d:
#                 for val in predictions :
#                     d.write(str(val)+'\n')
                

#             rows = zip(angry_0,disgust_1,fear_2,happy_3,sad_4,surprise_5,neutral_6)

#             import csv
#             with open("static/js/db/prob.csv", "w") as d:
#                 writer = csv.writer(d)
#                 for row in rows:
#                     writer.writerow(row)
            

#             with open("static/js/db/prob_tot.csv", "a") as d:
#                 writer = csv.writer(d)
#                 for row in rows:
#                     writer.writerow(row)
            
#         #     K.clear_session()
#         #     break

#     video_capture.release()

#     _emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
#     print(predictions)
#     predictions_name = [_emotion.get(emotion) for emotion in predictions]

#     all_predictions = {
#         'predictions':predictions_name,
#         'angry_0':angry_0,
#         'disgust_1':disgust_1,
#         'fear_2':fear_2,
#         'happy_3':happy_3,
#         'sad_4':sad_4,
#         'surprise_5':surprise_5,
#         'neutral_6':neutral_6
#     }

#     return all_predictions


