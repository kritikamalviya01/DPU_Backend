import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import csv
import time

def facial_emotion(video_path):
    """
    Process a video file to predict facial emotions.
    Returns a dictionary of predictions and associated probabilities.
    """
    # Initialize variables
    model = None
    face_detector = None
    predictor_landmarks = None
    predictions = []
    emotion_data = {}

    # Load the model and other resources
    try:
        model = load_model('Models/video.h5', compile=False)
        face_detector = dlib.get_frontal_face_detector()
        predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")
    except Exception as e:
        return {"error": f"Failed to load model or face detector: {str(e)}"}

    # Check if video exists
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        return {"error": "Could not open video file."}

    # Define input shape and classes
    shape_x, shape_y = 48, 48
    n_classes = 7
    emotion_data = {i: [] for i in range(n_classes)}

    # Timer setup
    max_time = 15
    start_time = time.time()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit if no frames are left

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 1)

        for rect in rects:
            # Get face coordinates and landmarks
            try:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face = gray[y:y + h, x:x + w]

                shape = predictor_landmarks(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Zoom and preprocess the face image
                face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
                face = face.astype(np.float32) / float(face.max())
                face = np.reshape(face.flatten(), (1, shape_x, shape_y, 1))

                # Make emotion prediction
                prediction = model.predict(face)
                for i in range(n_classes):
                    emotion_data[i].append(prediction[0][i].astype(float))
                predictions.append(np.argmax(prediction))

            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue  # Skip to the next face

        # Save frames as images if needed (optional)
        frame_filename = f"frames/frame_{int(time.time())}.jpg"
        cv2.imwrite(frame_filename, frame)  # Save the frame as an image

    # Release video capture
    video_capture.release()

    # Map predictions to emotion names
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    predictions_names = [emotion_mapping.get(emotion) for emotion in predictions]

    # Prepare the response data
    all_predictions = {
        'predictions': predictions_names,
        **{f'prob_{emotion_mapping[i]}': emotion_data[i] for i in range(n_classes)}
    }

    # Log predictions to a CSV file
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Prediction", "Probabilities"])
        for i, prediction in enumerate(predictions):
            writer.writerow([i, prediction, emotion_data[prediction]])

    return all_predictions






# =============================================================================================

# import os
# import cv2
# import numpy as np
# import dlib
# from imutils import face_utils
# from tensorflow.keras.models import load_model
# from scipy.ndimage import zoom
# import csv
# import time

# def facial_emotion(video_path):
#     """
#     Process a video file to predict facial emotions.
#     Returns a dictionary of predictions and associated probabilities.
#     """
#     # Initialize variables
#     model = None
#     face_detector = None
#     predictor_landmarks = None
#     predictions = []
#     emotion_data = {}

#     # Load the model and other resources
#     try:
#         model = load_model('Models/video.h5',compile=False)
#         face_detector = dlib.get_frontal_face_detector()
#         predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")
#     except Exception as e:
#         return {"error": f"Failed to load model or face detector: {str(e)}"}
#     # try:
#     #     model = load_model('Models/video.h5')
#     #     face_detector = dlib.get_frontal_face_detector()
#     #     predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")
#     # except Exception as e:
#     #     return {"error": f"Failed to load model or face detector: {str(e)}"}

#     # Video capture
#     # video_path = os.path.join('recorded_video', filename)
#     if not os.path.exists(video_path):
#         return {"error": f"Video file not found: {video_path}"}

#     video_capture = cv2.VideoCapture(video_path)

#     # Check if video opened successfully
#     if not video_capture.isOpened():
#         return {"error": "Could not open video file."}

#     # Define input shape and classes
#     shape_x, shape_y = 48, 48
#     n_classes = 7
#     emotion_data = {i: [] for i in range(n_classes)}

#     # Timer setup
#     max_time = 15
#     start_time = time.time()

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break  # Exit if no frames are left

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = face_detector(gray, 1)

#         for rect in rects:
#             # Get face coordinates and landmarks
#             try:
#                 (x, y, w, h) = face_utils.rect_to_bb(rect)
#                 face = gray[y:y + h, x:x + w]

#                 shape = predictor_landmarks(gray, rect)
#                 shape = face_utils.shape_to_np(shape)

#                 # Zoom and preprocess the face image
#                 face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
#                 face = face.astype(np.float32) / float(face.max())
#                 face = np.reshape(face.flatten(), (1, shape_x, shape_y, 1))
#                 # Zoom and preprocess the face image
#                 face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
#                 face = face.astype(np.float32) / float(face.max())
#                 face = np.reshape(face.flatten(), (1, shape_x, shape_y, 1))

#                 # Make emotion prediction
#                 prediction = model.predict(face)
#                 for i in range(n_classes):
#                     emotion_data[i].append(prediction[0][i].astype(float))
#                 predictions.append(np.argmax(prediction))

#             except Exception as e:
#                 print(f"Error processing face: {str(e)}")
#                 continue  # Skip to the next face

#         # Display the frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release video capture and close windows
#     video_capture.release()
#     cv2.destroyAllWindows()

#     # Map predictions to emotion names
#     emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#     predictions_names = [emotion_mapping.get(emotion) for emotion in predictions]

#     # Prepare the response data
#     all_predictions = {
#         'predictions': predictions_names,
#         **{f'prob_{emotion_mapping[i]}': emotion_data[i] for i in range(n_classes)}
#     }

#     return all_predictions


# =====================================================================================================================



# import os
# import cv2
# import numpy as np
# import dlib
# from imutils import face_utils
# from tensorflow.keras.models import load_model
# from scipy.ndimage import zoom
# import csv
# import time

# def facial_emotion(filename):
#     """
#     Process a video file to predict facial emotions.
#     Returns a dictionary of predictions and associated probabilities.
#     """
    
#     # Load the model and other resources
#     model = load_model('Models/video.h5')
#     face_detector = dlib.get_frontal_face_detector()
#     predictor_landmarks = dlib.shape_predictor("Models/face_landmarks.dat")

#     # Video capture
#     video_path = os.path.join('recorded_video', filename)
#     video_capture = cv2.VideoCapture(video_path)

#     # Check if video opened successfully
#     if not video_capture.isOpened():
#         return {"error": "Could not open video file."}

#     # Define input shape and classes
#     shape_x, shape_y = 48, 48
#     n_classes = 7
#     predictions = []
#     emotion_data = {i: [] for i in range(n_classes)}

#     # Timer setup
#     max_time = 15
#     start_time = time.time()

#     while video_capture.isOpened():
#         ret, frame = video_capture.read()
#         if not ret:
#             break  # Exit if no frames are left

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = face_detector(gray, 1)

#         for rect in rects:
#             # Get face coordinates and landmarks
#             (x, y, w, h) = face_utils.rect_to_bb(rect)
#             face = gray[y:y + h, x:x + w]

#             shape = predictor_landmarks(gray, rect)
#             shape = face_utils.shape_to_np(shape)

#             # Zoom and preprocess the face image
#             face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
#             face = face.astype(np.float32) / float(face.max())
#             face = np.reshape(face.flatten(), (1, shape_x, shape_y, 1))

#             # Make emotion prediction
#             prediction = model.predict(face)
#             for i in range(n_classes):
#                 emotion_data[i].append(prediction[0][i].astype(float))
#             predictions.append(np.argmax(prediction))

#         # Display the frame
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Write results to files after reaching max time
#         # if time.time() - start_time > max_time:
#         #     with open("static/js/db/histo_perso.txt", "w") as d:
#         #         d.write("density\n")
#         #         d.writelines(f"{val}\n" for val in predictions)

#         #     with open("static/js/db/histo.txt", "a") as d:
#         #         d.writelines(f"{val}\n" for val in predictions)

#         #     with open("static/js/db/prob.csv", "w") as d:
#         #         writer = csv.writer(d)
#         #         writer.writerows(zip(*[emotion_data[i] for i in range(n_classes)]))

#         #     with open("static/js/db/prob_tot.csv", "a") as d:
#         #         writer = csv.writer(d)
#         #         writer.writerows(zip(*[emotion_data[i] for i in range(n_classes)]))

#     # Release video capture and close windows
#     video_capture.release()
#     cv2.destroyAllWindows()

#     # Map predictions to emotion names
#     emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#     predictions_names = [emotion_mapping.get(emotion) for emotion in predictions]

#     # Prepare the response data
#     all_predictions = {
#         'predictions': predictions_names,
#         **{f'prob_{emotion_mapping[i]}': emotion_data[i] for i in range(n_classes)}
#     }

#     return all_predictions


