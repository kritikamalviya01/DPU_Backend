import threading
from flask import Flask, jsonify, request, render_template
import os
import requests
from flask_cors import CORS
from dotenv import load_dotenv 
from pymongo import MongoClient
from bson.objectid import ObjectId
import logging


from src.speech_to_text import speech_to_text
from src.facial_emotion import facial_emotion
from src.speech_emotion1 import speechEmotionRecognition
from src.compare import compare

# from utils.database.pre_defined_questions import retrive_preDefinedQA
# from utils.database.candidates import *
from utils.extract_audio import extract_audio
# from utils.s3_storage import upload_to_s3, download_file_from_s3

load_dotenv()
os.environ['FLASK_ENV'] = 'development'
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME")

client = MongoClient(mongo_uri)
db = client[mongo_db_name]
interviews_collection = db["interviews"]
questions_collection = db['questions']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'recorded_video'
AUDIO_FOLDER = 'recorded_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
    
def get_question_details(questionId):
    """Fetch question details by questionId."""
    return questions_collection.find_one({'_id': ObjectId(questionId)})

def download_video(video_url, save_path):
    """Download the video from a URL and save it to the specified path."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return {"success": True, "path": save_path}
    except Exception as e:
        return {"error": f"Failed to download video: {str(e)}"}
    
def update_stage(interviewId, questionId, field, stage, data=None):
    """Update the stage and optional data for a specific field in the database."""
    filter_query = {
        "_id": ObjectId(interviewId),
        "responses.questionId": ObjectId(questionId)
    }
    
    update_fields = {f"responses.$.{field}.stage": stage}
    if data is not None:
        update_fields[f"responses.$.{field}.data"] = data

    result = interviews_collection.update_one(filter_query, {"$set": update_fields})
    
    if result.matched_count == 0:
        print("No matching document or response found for the given interviewId and questionId.")
    else:
        print(f"Stage updated to '{stage}' for field '{field}'.")
    
    return result


def process_video(file_path, interviewId, questionId):
    """Background task for processing video."""
    try:
        print("Processing video:", file_path)

        # Extract audio from the video
        audio_file = extract_audio(file_path)
        if isinstance(audio_file, dict) and "error" in audio_file:
            print("Error in audio extraction:", audio_file["error"])
            raise Exception(audio_file["error"])

        # Ensure audio file exists before proceeding
        audio_file_path = audio_file
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        print("Audio file path:", audio_file_path)

        # Start analysis in separate threads
        # threading.Thread(target=facial_emotions_analysis, args=(file_path, interviewId, questionId)).start()
        # threading.Thread(target=analyze_speech_emotion, args=(audio_file_path, interviewId, questionId)).start()
        threading.Thread(target=analyze_speech_to_text, args=(audio_file_path, interviewId, questionId)).start()

        print(f"Processing started for Interview ID: {interviewId}, Question ID: {questionId}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")

def facial_emotions_analysis(file_path, interviewId, questionId):
    """Background task for analyzing facial emotions."""
    # Log a message indicating the background task has started
    logging.info("Background task for analyzing facial emotions.")
    
    # Set the stage to 'started' before processing
    update_stage(interviewId, questionId, "facialEmotions", "started")
    
    try:
        # Call the facial emotion analysis function
        prediction = facial_emotion(file_path)
        
        # Check if the prediction contains an error
        if 'error' in prediction:
            raise Exception(f"Error analyzing facial emotions: {prediction['error']}")
        
        # Log the predictions
        logging.info("Predictions: %s", prediction)
        
        # If successful, update the stage to 'success' with prediction data
        update_stage(interviewId, questionId, "facialEmotions", "success", prediction)
        
        logging.info("Facial emotion analysis succeeded.")
    
    except Exception as e:
        # On failure, update the stage to 'failed'
        update_stage(interviewId, questionId, "facialEmotions", "failed")
        
        # Log the error
        logging.error("Error analyzing facial emotions: %s", str(e))
    
    finally:
        # Cleanup: Remove the file after processing
        # file_path = os.path.join('recorded_video' filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info("Removed file: %s", file_path)
        else:
            logging.warning("File not found: %s", file_path)
            
def analyze_speech_emotion(audio_file_path, interviewId, questionId):
    """Background task for analyzing speech emotions."""
    print("Starting background task for analyzing speech emotions.")
    
    # Set the initial stage to 'started'
    update_stage(interviewId, questionId, "speechEmotions", "started")
    
    try:
        # Load the speech emotion recognition model
        model_path = os.path.join('Models', 'audio.hdf5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Initialize the speech emotion recognition model
        SER = speechEmotionRecognition(model_path)
        
        # Perform speech emotion analysis on the provided audio file
        emotions, _ = SER.predict_emotion_from_file(audio_file_path)

        if not emotions:
            raise ValueError("No emotions detected in the audio file.")

        # Calculate the major emotion and distribution
        major_emotion = max(set(emotions), key=emotions.count)
        emotion_dist = {
            emotion: int(100 * emotions.count(emotion) / len(emotions)) for emotion in set(emotions)
        }

        # Prepare the data to be sent in the update
        data = {
            'emotions': emotions,
            'major_emotion': major_emotion,
            'emotion_dist': emotion_dist
        }

        print(f"Speech emotion analysis successful: {data}")
        update_stage(interviewId, questionId, "speechEmotions", "success", data)

    except FileNotFoundError as fnf_error:
        error_message = f"File not found: {str(fnf_error)}"
        print(error_message)
        update_stage(interviewId, questionId, "speechEmotions", "failed")

    except ValueError as val_error:
        error_message = f"Value error: {str(val_error)}"
        print(error_message)
        update_stage(interviewId, questionId, "speechEmotions", "failed")

    except Exception as e:
        error_message = f"Unexpected error analyzing speech emotions: {str(e)}"
        print(error_message)
        update_stage(interviewId, questionId, "speechEmotions", "failed")
        
def analyze_speech_to_text(audio_file_path, interviewId, questionId):
    """Convert speech to text and analyze the comparison."""
    print("Starting speech-to-text conversion.")

    # Set the initial stage to 'started'
    update_stage(interviewId, questionId, "transcript", "started")

    try:
        # Perform speech-to-text conversion
        text, error = speech_to_text(audio_file_path)

        if error:
            raise Exception(error)

        if not text or len(text.strip()) == 0:
            raise ValueError("Transcription resulted in empty text.")

        # Log and update the successful transcript result
        print(f"Transcription successful. Transcribed text: {text}")
        update_stage(interviewId, questionId, "transcript", "success", text)

        # Fetch question details from the database
        print("Fetching question details from the database.")
        question_details = get_question_details(questionId)

        if not question_details:
            raise ValueError(f"No question found for questionId: {questionId}")

        # Extract the correct answer from question details
        correct_answer = question_details.get('answer')

        if not correct_answer:
            raise ValueError(f"No answer found in question details for questionId: {questionId}")

        # Start the comparison analysis in a separate thread
        print("Starting comparison analysis in a separate thread.")
        threading.Thread(
            target=perform_comparison_analysis,
            args=(interviewId, questionId, text, correct_answer)
        ).start()

    except FileNotFoundError as fnf_error:
        error_message = f"Audio file not found: {str(fnf_error)}"
        print(error_message)
        update_stage(interviewId, questionId, "transcript", "failed")
        update_stage(interviewId, questionId, "comparisonScore", "failed")

    except ValueError as val_error:
        error_message = f"Value error: {str(val_error)}"
        print(error_message)
        update_stage(interviewId, questionId, "transcript", "failed")
        update_stage(interviewId, questionId, "comparisonScore", "failed")

    except Exception as e:
        error_message = f"Error during speech-to-text conversion: {str(e)}"
        print(error_message)
        update_stage(interviewId, questionId, "transcript", "failed")
        update_stage(interviewId, questionId, "comparisonScore", "failed")

def perform_comparison_analysis(interviewId, questionId, transcript, correct_answer):
    """Perform comparison analysis and update the comparison score."""
    try:
        update_stage(interviewId, questionId, "comparisonScore", "started")

        # Calculate the comparison score
        comparison_score = compare(transcript, correct_answer)
        
        comparison_score = float(comparison_score)

        print(f"Comparison successful. Score: {comparison_score}%")
        update_stage(
            interviewId,
            questionId,
            "comparisonScore",
            "success",
            comparison_score
        )

    except Exception as e:
        error_message = f"Error during comparison analysis: {str(e)}"
        print(error_message)
        update_stage(interviewId, questionId, "comparisonScore", "failed")



@app.route("/api/python/process-video", methods=["POST"])
def process_video_api():
    """API endpoint to process a video."""
    try:
        data = request.json
        interviewId = data.get("interviewId")
        questionId = data.get("questionId")
        videoUrl = data.get("videoUrl")

        # Validate input parameters
        if not interviewId or not questionId or not videoUrl:
            return jsonify({"error": "Missing required parameters"}), 400

        # Generate a unique filename for the video
        video_filename = f"{interviewId + questionId}.mp4"
        save_path = os.path.join(UPLOAD_FOLDER, video_filename)

        # Download the video
        download_result = download_video(videoUrl, save_path)
        if "error" in download_result:
            return jsonify({"error": download_result["error"]}), 400

        # Start video processing in a background thread
        threading.Thread(target=process_video, args=(save_path, interviewId, questionId)).start()

        return jsonify({"message": "Video processing started successfully", "interviewId": interviewId, "questionId": questionId}), 202

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



    # -----------------------------------------------------------------------------

#  
# def process_audio(file_path, objectId, filename):
#     """Background task for processing video."""
#     print('"""Background task for processing video."""')
#     try:
#         print('file path', file_path)
       
#         if not os.path.exists(file_path):
#             print(f"Audio file not found: {file_path}")
#             raise FileNotFoundError(f"Audio file could not be found: {file_path}")

#         threading.Thread(target=analyze_speech_emotion_in_background, args=(file_path, objectId)).start()
#         threading.Thread(target=analyze_speech_to_text, args=(file_path, objectId)).start()

#     except Exception as e:
#         print(f"Error processing video: {str(e)}")

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload for video or audio."""
#     # Retrieve uploaded video and audio files
#     uploaded_video = request.files.get('video')
#     uploaded_audio = request.files.get('audio')

#     # Check if neither file was provided
#     if not uploaded_video and not uploaded_audio:
#         return jsonify({"error": "No file provided"}), 400

#     # Handle video file upload
#     if uploaded_video:
#         filename = uploaded_video.filename
#         if not filename:
#             return jsonify({"error": "No filename provided for video."}), 400
#         print('filename ', filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         print('Video file path:', file_path)

#         try:
#             uploaded_video.save(file_path)
#             objectId, _ = create_null_document()
#             video_url = upload_to_s3(file_path, filename)
#             result = update_into_db(objectId, name=filename[:-4], type="Video", title=filename, url=video_url)
#             threading.Thread(target=process_video, args=(file_path, objectId, filename)).start()

#             return jsonify({"message": "Video uploaded successfully.", "filename": filename, "data" : result}), 200

#         except Exception as e:
#             return jsonify({"error": f"Failed to upload video: {str(e)}"}), 500

#     # Handle audio file upload
#     if uploaded_audio:
#         filename = uploaded_audio.filename
#         if not filename:
#             return jsonify({"error": "No filename provided for audio."}), 400
        
#         file_path = os.path.join(AUDIO_FOLDER, filename)
#         print('Audio file path:', file_path)

#         try:
#             uploaded_audio.save(file_path)
#             objectId, _ = create_null_document()
#             audio_url = upload_to_s3(file_path, filename)
#             result = update_into_db(objectId, name=filename[:-4], type="Audio", title=filename, url=audio_url)
#             threading.Thread(target=process_audio, args=(file_path, objectId, filename)).start()

#             return jsonify({"message": "Audio uploaded successfully.", "filename": filename, "data" : result}), 200

#         except Exception as e:
#             return jsonify({"error": f"Failed to upload audio: {str(e)}"}), 500

# def update_stage(objectId, field, stage, data=None):
#     """Update the stage and optional data for a specific field in the database."""
#     update_fields = {f"{field}.stage": stage}
#     if data is not None:
#         update_fields[f"{field}.data"] = data
#     result = updateFieldsIntoDB(objectId, update_fields)
#     return result

# def analyze_facial_emotion_in_background(filename, objectId):
#     """Background task for analyzing facial emotions."""
#     print("""Background task for analyzing facial emotions.""")
#     update_stage(objectId, "facialEmotions", "started")  # Set stage to started
#     try:
#         prediction = facial_emotion(filename)
#         # Check if the prediction contains an error
#         if 'error' in prediction:
#             # update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
#             # print(f"Error analyzing facial emotions: {prediction['error']}")
#             # return   # Return the error response
#             raise Exception("Error analyzing facial emotions: {prediction['error']}")

#         print("facial emotion success")
#         result = update_stage(objectId, "facialEmotions", "success", prediction)  # Set stage to success
#         return result
#     except Exception as e:
#         update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
#         print(f"Error analyzing facial emotions: {str(e)}")
#     finally:
#         # Remove the file after processing
#         file_path = os.path.join('recorded_video', filename)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {filename}")
#         else:
#             print(f"File not found: {filename}")

# def analyze_speech_emotion_in_background(audio_file, objectId):
#     """Background task for analyzing speech emotions."""
#     print("""Background task for analyzing speech emotions.""")
#     update_stage(objectId, "speechEmotions", "started")  # Set stage to started
#     try:
#         model = os.path.join('Models', 'audio.hdf5')
#         SER = speechEmotionRecognition(model)
#         emotions, _ = SER.predict_emotion_from_file(audio_file)

#         major_emotion = max(set(emotions), key=emotions.count)
#         emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
        
#         data = {
#             'emotions': emotions,
#             'major_emotion': major_emotion,
#             'emotion_dist': emotion_dist
#         }
#         print("success speech emotion")
#         update_stage(objectId, "speechEmotions", "success", data)  # Set stage to success

#     except Exception as e:
#         update_stage(objectId, "speechEmotions", "failed")  # Set stage to failed
#         print(f"Error analyzing speech emotions: {str(e)}")

# def analyze_speech_to_text(audio_file, objectId):
#     """Convert speech to text."""
#     print("Starting speech-to-text conversion.")
#     update_stage(objectId, "transcript", "started")  # Set stage to started

#     try:
#         text, error = speech_to_text(audio_file)  # Get transcription result
#         if error:  # Check if there's an error message
#             raise Exception(error)

#         print("Transcription successful. Transcribed text:", text)
#         update_stage(objectId, "transcript", "success", text)  # Set stage to success
        
#         # Start the comparison in a separate thread
#         threading.Thread(target=analyze_compare, args=(audio_file, objectId, text)).start()

#     except Exception as e:
#         update_stage(objectId, "transcript", "failed")  # Set stage to failed
#         update_stage(objectId, "comparePercentage", "failed")  # Set compare percentage stage to failed
#         print(f"Error during speech-to-text conversion: {str(e)}")



# def analyze_compare(filename, objectId):
#     """Perform comparison analysis."""
#     update_stage(objectId, "comparePercentage", "started")  # Set stage to started
#     try:
#         compare_score = compare(filename[:-4])  # Assuming compare() accepts a filename
#         update_stage(objectId, "comparePercentage", "success", float(compare_score))  # Set stage to success
#     except Exception as e:
#         update_stage(objectId, "comparePercentage", "failed")  # Set stage to failed
#         print(f"Error during comparison analysis: {str(e)}")

# def analyze_compare(file_path, objectId, text):
#     """Analyze the comparison of speech text with predefined questions."""
#     # try:
#     #     # Retrieve the speech text from the database
#     #     # file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
#     #     speech_text, _ = speech_to_text(file_path)  # Get the transcribed text
#     # except Exception as e:
#     #     print(f"Error during speech-to-text conversion: {str(e)}")
#     #     update_stage(objectId, "comparePercentage", "failed")
#     #     return

#     try:
#         # Define parameters for retrieving questions
#         level = "beginner"  # You can modify this as needed
#         tech = "nodejs"  # You can modify this as needed
#         answer = 1  # Example answer, adjust as necessary

#         # Retrieve predefined questions
#         documents = retrive_preDefinedQA(level, tech, answer)
        
#         if not documents:
#             print("No documents retrieved for comparison.")
#             update_stage(objectId, "comparePercentage", "failed")
#             return
        
#         # Combine answers from retrieved questions
#         combined_answers = ' '.join([doc['answer'] for doc in documents if 'answer' in doc])
#     except Exception as e:
#         print(f"Error during question retrieval: {str(e)}")
#         update_stage(objectId, "comparePercentage", "failed")
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {file_path}")
#         return

#     try:
#         # Perform comparison
#         compare_score = compare(text, combined_answers)  # Assuming compare function can take both texts
#         update_stage(objectId, "comparePercentage", "success", float(compare_score))  # Set stage to success
#         # update_into_db(objectId=objectId, compare_percentage=float(compare_score))
#     except Exception as e:
#         print(f"Error during comparison: {str(e)}")
#         update_stage(objectId, "comparePercentage", "failed")
#     finally:
#         # Remove the file after processing
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {file_path}")
#         else:
#             print(f"File not found: {file_path}")


# @app.route('/retrieve', methods=['GET'])
# def retrieve_questions():
#     level = request.args.get('level')
#     tech = request.args.get('tech')
#     answer = request.args.get("answer")
#     documents = retrive_preDefinedQA(level, tech, int(answer))
#     return jsonify(documents)

# @app.route('/get-facial-emotions/<id>', methods=['GET'])
# def getFacialEmotions(id):
#     if not id:
#         return jsonify({"error": "Please provide id"}), 400
    
#     # Step 1: Get interview details
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
#     if error_message:
#         return jsonify({"error": error_message}), status_code

#     if interview_details['type'] == 'Audio':
#          return jsonify({"error": "This is an audio interview cannot process video"}), 400

#     # Step 2: Extract video URL from interview details
#     video_url = interview_details["VideoDetails"]["url"]  # Adjust based on your data structure
#     if not video_url:
#         return jsonify({"error": "Video link not found in interview details"}), 404

#     # Step 3: Download the video
#     try:
#         # response = requests.get(video_url)
#         downloaded_file_name = download_file_from_s3(video_url, "Video")
#         print("DOWNLOADED PATH ", downloaded_file_name)
#         if downloaded_file_name:
#         # # response.raise_for_status()  # Raise an error for bad responses
#         #     # video_file_path = os.path.join(UPLOAD_FOLDER, f"{id}.mp4")  # Save with the interview ID
#         #     with open(video_file_path, 'wb') as f:
#         #         f.write(response.content)
#             print(f"Downloaded video to: {downloaded_file_name}")
#         else :
#             raise Exception("Unable to download requested file")

#     except Exception as e:
#         return jsonify({"error": f"Failed to download video: {str(e)}"}), 500

#     # Step 4: Perform facial recognition analysis
#     try:
#         update_stage(id, "facialEmotions", "started")  # Set stage to started

#         # Analyze facial emotions in a separate thread
#         result = analyze_facial_emotion_in_background(downloaded_file_name, id)
#         if 'error' in result:
#             update_stage(id, "facialEmotions", "failed")  # Set stage to failed
#             print(f"Error analyzing facial emotions: {result['error']}")
#             return jsonify({"error": f"Facial recognition analysis failed: {str(result['error'])}"}), 500  # Return the error response

#         return jsonify({"message": "Facial recognition analysis started successfully", "result": result}), 200

#     except Exception as e:
#         return jsonify({"error": f"Facial recognition analysis failed: {str(e)}"}), 500

# @app.route('/interview/<id>', methods=['GET'])
# def get_interview(id):
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
    
#     if error_message:
#         return jsonify({"error": error_message}), status_code
    
#     return { "message" : "successfully fetched interview details!", "data" : interview_details}, 200

@app.route('/testroute', methods=['GET'])
def testPing():
     return { "message" : "successfully fetched interview details new!"}, 200


if __name__ == "__main__":
    app.run(port=8080,debug=True)
















# import threading
# from flask import Flask, jsonify, request, render_template, send_file
# import os
# import requests
# import subprocess
# from flask_cors import CORS
# import pandas as pd

# from src.speech_to_text import speech_to_text
# from src.facial_emotion import facial_emotion
# from src.speech_emotion1 import speechEmotionRecognition
# from src.compare import compare

# from utils.database.candidates import *
# from utils.database.pre_defined_questions import retrive_preDefinedQA
# from utils.extract_audio import extract_audio
# from utils.s3_storage import upload_to_s3, download_file_from_s3

# app = Flask(__name__)
# CORS(app)

# # @app.route('/upload')
# # def upload_file():
# #    return render_template('upload.html')

# # @app.route('/recorded_video',methods=['GET','POST'])
# # def recorded_video():
# #     if request.method == 'POST':
# #         f = request.files['video']
# #         global filename
# #         filename = f.filename
# #         f.save(os.path.join('recorded_video',filename))
# #         return 'File Save Successfully'
# #     elif request.method == 'GET':
# #         return send_file(os.path.join('recorded_video',filename))

# UPLOAD_FOLDER = 'recorded_video'
# RECORDED_AUDIO = 'recorded_audio'
# TMP = 'tmp'

# for folder in [UPLOAD_FOLDER, RECORDED_AUDIO, TMP]:
#     os.makedirs(folder, exist_ok=True)

# # @app.route('/upload', methods=['POST'])
# # def upload_video():
# #     f = request.files['video']
# #     if not f:
# #         return jsonify({"error": "No file provided"}), 400
    

# #     global filename
# #     filename = f.filename
# #     with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
# #         temp_file.write(f.read())
# #         temp_file.seek(0)

# #     # # print(f.read())
# #     # # video_data = BytesIO(f.read())
# #     # video_clip = VideoFileClip(temp_file.name)
# #     # video_clip.write_videofile(os.path.join(UPLOAD_FOLDER, filename), codec="libx264")
# #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# #     f.save(file_path)

# #     audio_file = extract_audio(file_path)
# #     if isinstance(audio_file, dict) and "error" in audio_file:
# #         return jsonify({"error": audio_file["error"]})
    
# #     return jsonify({"message": "File saved successfully", "filename": filename}), 200

# def update_stage(objectId, field, stage, data=None):
#     """Update the stage and optional data for a specific field in the database."""
#     update_fields = {f"{field}.stage": stage}
#     if data is not None:
#         update_fields[f"{field}.data"] = data
#     result = update_index(objectId, update_fields)
#     return result

# audio_event = threading.Event()
# transcribe_event = threading.Event()
# compare_event = threading.Event()

# def process_video(file_path, objectId, filename):
#     """Background task for processing video"""
    

#     audio_file = extract_audio(file_path)
#     audio_event.set()
#     transcribe_event.set()
#     if isinstance(audio_file, dict) and "error" in audio_file:
#         return jsonify({"error": audio_file["error"]})
    
# def process_audio(file_path, objectId, filename):
#     object_name = f"{filename[:-4]}_{str(objectId)}.mp3"
#     audio_url = upload_to_s3(file_path, object_name)
#     update_into_db(objectId, name=filename[:-4], type="Audio", title=object_name, url=audio_url)
#     audio_event.set()
#     transcribe_event.set()

# ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
# ALLOWED_AUDIO_EXTENSIONS = {'mp3'}

# def allowed_file(filename, allowed_extensions):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     video = request.files.get('video')
#     audio = request.files.get("audio")
#     if not video and not audio:
#         return jsonify({"error": "No file provided"}), 400
#     if video and audio:
#         return jsonify({"error": "Upload either a video or an audio file, not both."}), 400

#     global filename
#     global objectId

#     if video:
#         filename = video.filename
#         if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
#             return jsonify({"error": "Invalid video file type. Only .mp4 allowed."}), 415
#         os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#         os.makedirs(RECORDED_AUDIO, exist_ok=True)
#         video_file_path = os.path.join(UPLOAD_FOLDER, filename)
#         # audio_file_path = os.path.join("recorded_audio", filename)
#         video.save(video_file_path)
#         audio_file_path = extract_audio(video_file_path)
#         objectId, null_document = create_null_document(type="Video")
#         video_object_name = f"{str(objectId)}/{filename[:-4]}.mp4"
#         audio_object_name = f"{str(objectId)}/{filename[:-4]}.mp3"
#         video_url = upload_to_s3(video_file_path, video_object_name)
#         audio_url = upload_to_s3(audio_file_path, audio_object_name)
#         result = update_into_db(objectId, name=filename[:-4], type="Video", title=filename[:-4], 
#                                 video_url=video_object_name, audio_url=audio_object_name)
#         # Run video processing in a separate thread
#         # threading.Thread(target=process_video, args=(video_file_path, objectId, filename,)).start()
#         return jsonify({"message": "Video uploaded successfully.", "filename": filename, "data" : result}), 200
#         update_fields = {}

#         try:
#             update_fields["facialEmotions.stage"] = "Processing"
#             threading.Thread(target=analyze_facial_emotion_in_background, args=(filename,)).start()
#         except:
#             update_fields["facialEmotions.stage"] = "Failed"

#         try:
#             update_fields["speechEmotions.stage"] = "Processing"
#             threading.Thread(target=analyze_speech_emotion_in_background, args=(filename,)).start()
#         except:
#             update_fields["speechEmotions.stage"] = "Failed"

#         try:
#             update_fields["transcript.stage"] = "Processing"
#             threading.Thread(target=analyze_speech_to_text, args=(filename,)).start()
#         except:
#             update_fields["transcript.stage"] = "Failed"

#         try:
#             update_fields["comparePercentage.stage"] = "Processing"
#             threading.Thread(target=analyze_compare, args=(filename,)).start()
#         except:
#             update_fields["comparePercentage.stage"] = "Failed"

#         update_index(objectId, update_fields)

#     elif audio:
#         filename = audio.filename
#         if not allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
#             return jsonify({"error": "Invalid audio file type. Only .mp3 allowed."}), 415

#         os.makedirs(RECORDED_AUDIO, exist_ok=True)
#         file_path = os.path.join("recorded_audio", filename)
#         audio.save(file_path)
#         objectId, null_document = create_null_document(type="Audio")
#         threading.Thread(target=process_audio, args=(file_path, objectId, filename,)).start()
#         update_fields = {}

#         try:
#             update_fields["speechEmotions.stage"] = "Processing"
#             threading.Thread(target=analyze_speech_emotion_in_background, args=(filename,)).start()
#         except:
#             update_fields["speechEmotions.stage"] = "Failed"

#         try:
#             update_fields["transcript.stage"] = "Processing"
#             threading.Thread(target=analyze_speech_to_text, args=(filename,)).start()
#         except:
#             update_fields["transcript.stage"] = "Failed"

#         try:
#             update_fields["comparePercentage.stage"] = "Processing"
#             threading.Thread(target=analyze_compare, args=(filename,)).start()
#         except:
#             update_fields["comparePercentage.stage"] = "Failed"

#         update_index(objectId, update_fields)

#     return jsonify({"message": f"File saved successfully,", "filename": filename}), 200


# @app.route('/recorded_video/<filename>', methods=['GET'])
# def recorded_video(filename):
#     filename = filename
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     if os.path.exists(file_path):
#         return send_file(file_path), 200
#     else:
#         return jsonify({"error": "File not found"}), 404
    
# # def analyze_facial_emotion_in_background(filename):
# #     """Background task for analyzing facial emotions"""
# #     update_fields = {}
# #     prediction = facial_emotion(filename)
# #     update_fields["facialEmotions.stage"] = "Completed"
# #     update_index(objectId, update_fields)
# #     update_into_db(objectId=objectId, facial_emotions=prediction)

# def analyze_facial_emotion_in_background(filename, objectId):
#     """Background task for analyzing facial emotions."""
#     print("""Background task for analyzing facial emotions.""")
#     update_stage(objectId, "facialEmotions", "started")  # Set stage to started
#     try:
#         prediction = facial_emotion(filename)
#         # Check if the prediction contains an error
#         if 'error' in prediction:
#             update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
#             # print(f"Error analyzing facial emotions: {prediction['error']}")
#             # return   # Return the error response
#             raise Exception("Error analyzing facial emotions: {prediction['error']}")

#         print("facial emotion success")
#         result = update_stage(objectId, "facialEmotions", "success", prediction)  # Set stage to success
#         return result
#     except Exception as e:
#         update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
#         print(f"Error analyzing facial emotions: {str(e)}")
#     finally:
#         # Remove the file after processing
#         file_path = os.path.join('recorded_video', filename)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {filename}")
#         else:
#             print(f"File not found: {filename}")

# @app.route('/get_facial_emotions', methods=['GET'])
# def get_facial_emotion():
#     filename = request.args.get('filename')  # Get filename from request params
#     if not filename:
#         return jsonify({"error": "No filename provided"}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     if os.path.exists(file_path):
#         # Run facial emotion analysis in the background
#         threading.Thread(target=analyze_facial_emotion_in_background, args=(filename,)).start()
#         return jsonify({"message": "Facial emotion analysis started", "status_code": 200}), 200
#     else:
#         return jsonify({"error": "File not found", "status_code": 404}), 404

# def analyze_speech_emotion_in_background(filename, objectId):
#     """Background task for analyzing speech emotions"""
#     audio_event.wait()
#     audio_event.clear()
#     try:
#         file_path = os.path.join('recorded_audio', filename[:-1] + '3')
#         model = os.path.join('Models', 'audio.hdf5')
#         SER = speechEmotionRecognition(model)
#         emotions, timestamp = SER.predict_emotion_from_file(file_path)

#         major_emotion = max(set(emotions), key=emotions.count)
#         emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

#         data = {
#             'emotions': emotions,
#             'major_emotion': major_emotion,
#             'emotion_dist': emotion_dist
#         }
#         update_fields = {}
#         update_fields["speechEmotions.stage"] = "Success"
#         update_index(objectId, update_fields)
#         result = update_into_db(objectId=objectId, speech_emotions=data)
#         return result
#     except Exception as e:
#         update_stage(objectId, "speechEmotions", "failed")  # Set stage to failed
#         print(f"Error analyzing speech emotions: {str(e)}")
#     finally:
#         # Remove the file after processing
#         file_path = os.path.join('recorded_audio', filename)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {filename}")
#         else:
#             print(f"File not found: {filename}")

# @app.route('/get_speech_emotion', methods=['GET'])
# def get_speech_emotion():
#     filename = request.args.get('filename')  # Get filename from request params
#     if not filename:
#         return jsonify({"error": "No filename provided"}), 400

#     file_path = os.path.join('recorded_audio', filename[:-1] + '3')
#     if os.path.exists(file_path):
#         # Run speech emotion analysis in the background
#         threading.Thread(target=analyze_speech_emotion_in_background, args=(filename,)).start()
#         return jsonify({"message": "Speech emotion analysis started", "status_code": 200}), 200
#     else:
#         return jsonify({"error": "File not found", "status_code": 404}), 404
    
# def analyze_speech_to_text(filename, objectId):
#     transcribe_event.wait()
#     transcribe_event.clear()
#     file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
#     try:
#         if os.path.exists(file_path):
#             text, status_code = speech_to_text(file_path, filename[:-4])  # Exclude extension
#             update_fields = {}
#             update_fields["transcript.stage"] = "Success"
#             update_index(objectId, update_fields)
#             result = update_into_db(objectId=objectId, transcript=text)
#         compare_event.set()
#         return result
#     except Exception as e:
#         update_stage(objectId, "transcript", "failed")  # Set stage to failed
#         print(f"Error analyzing transcript: {str(e)}")
#     finally:
#         # Remove the file after processing
#         file_path = os.path.join('recorded_audio', filename)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Removed file: {filename}")
#         else:
#             print(f"File not found: {filename}")
    
# @app.route('/speech_to_text', methods=['GET'])
# def get_text():
#     filename = request.args.get('filename')  # Get filename from request params
#     if not filename:
#         return jsonify({"error": "No filename provided"}), 400
    
#     file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
#     if os.path.exists(file_path):
#         threading.Thread(target=analyze_speech_to_text, args=(filename,)).start()
#         return jsonify({"message": "Speech-to-text analysis started", "status_code": 200}), 200
#     else:
#         return jsonify({"error": "File not found", "status_code": 404}), 404

# def analyze_compare(filename, objectId):
#     compare_event.wait()
#     compare_event.clear()
#     try:
#         compare_score = compare(filename)  # Assuming `compare()` accepts a filename as a parameter
#         update_fields = {}
#         update_fields["comparePercentage.stage"] = "Completed"
#         update_index(objectId, update_fields)
#         result = update_into_db(objectId=objectId, compare_percentage=float(compare_score))
#         return result
#     except Exception as e:
#         update_stage(objectId, "comparePercentage", "failed")  # Set stage to failed
#         print(f"Error analyzing Comparison: {str(e)}")

# @app.route('/compare', methods=['GET', 'POST'])
# def compare_text():
#     filename = request.args.get('filename')  # Get filename from request params
#     if not filename:
#         return jsonify({"error": "No filename provided"}), 400
    
#     # Perform comparison logic
#     threading.Thread(target=analyze_compare, args=(filename,)).start()
#     return jsonify({"message": "Comparision analysis started", "status_code": 200}), 200

# @app.route('/retrieve', methods=['GET'])
# def retrieve_questions():
#     level = request.args.get('level')
#     tech = request.args.get('tech')
#     answer = request.args.get("answer")
#     documents = retrive_preDefinedQA(level, tech, int(answer))
#     return jsonify(documents)

# # @app.route('/delete_video/<filename>', methods=['DELETE'])
# # def delete_video(filename):
# #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# #     if os.path.exists(file_path):
# #         os.remove(file_path)
# #         return jsonify({"message": f"File {filename} deleted successfully", "status_code": 200}), 200
# #     else:
# #         return jsonify({"error": "File not found", "status_code": 404}), 404

# # @app.route('/get_facial_emotions',methods=['GET'])
# # def get_facial_emotion():
# #     prediction = facial_emotion(filename)
# #     return jsonify(prediction)

# # @app.route('/get_speech_emotion',methods=['GET'])
# # def get_speech_emotion():
# #     model = os.path.join('Models', 'audio.hdf5')
# #     SER = speechEmotionRecognition(model)
# #     audio = os.path.join('recorded_audio',filename[:-1]+'3')
# #     emotions, timestamp = SER.predict_emotion_from_file(audio)
# #     SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
# #     SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

# #     major_emotion = max(set(emotions), key=emotions.count)

# #     emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

# #     df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
# #     df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

# #     df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

# #     major_emotion_other = df_other.EMOTION.mode()[0]

# #     emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

# #     df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
# #     df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')

# #     data = {
# #         'emotions' : emotions,
# #         'major_emotion' : major_emotion,
# #         'emotion_dist' : emotion_dist,
# #         'major_emotion_other' : major_emotion_other,
# #         'emotion_dist_other' : emotion_dist_other
# #     }

# #     return jsonify(data)

# # @app.route('/speech_to_text',methods = ['GET'])
# # def get_text():
# #     text = speech_to_text(filename)
# #     return text

# # @app.route('/compare',methods = ['GET','POST'])
# # def compare_text():
# #     compare_score = compare()
# #     return str(round(compare_score,2) * 100)

# @app.route('/get-facial-emotions/<id>', methods=['GET'])
# def getFacialEmotions(id):
#     if not id:
#         return jsonify({"error": "Please provide id"}), 400
    
#     # Step 1: Get interview details
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
#     if error_message:
#         return jsonify({"error": error_message}), status_code

#     if interview_details['type'] == 'Audio':
#          return jsonify({"error": "This is an audio interview cannot process video"}), 400

#     # Step 2: Extract video URL from interview details
#     video_url = interview_details["uploadDetails"]["videoUrl"]  # Adjust based on your data structure
#     if not video_url:
#         return jsonify({"error": "Video link not found in interview details"}), 404

#     # Step 3: Download the video
#     try:
#         # response = requests.get(video_url)
#         downloaded_file_name = download_file_from_s3(video_url, "Video")
#         print("DOWNLOADED PATH ", downloaded_file_name)
#         if downloaded_file_name:
#         # # response.raise_for_status()  # Raise an error for bad responses
#         #     # video_file_path = os.path.join(UPLOAD_FOLDER, f"{id}.mp4")  # Save with the interview ID
#         #     with open(video_file_path, 'wb') as f:
#         #         f.write(response.content)
#             print(f"Downloaded video to: {downloaded_file_name}")
#         else :
#             raise Exception("Unable to download requested file")

#     except Exception as e:
#         return jsonify({"error": f"Failed to download video: {str(e)}"}), 500

#     # Step 4: Perform facial recognition analysis
#     try:
#         update_stage(id, "facialEmotions", "started")  # Set stage to started

#         # Analyze facial emotions in a separate thread
#         result = analyze_facial_emotion_in_background(downloaded_file_name, id)
#         if 'error' in result:
#             update_stage(id, "facialEmotions", "failed")  # Set stage to failed
#             print(f"Error analyzing facial emotions: {result['error']}")
#             return jsonify({"error": f"Facial recognition analysis failed: {str(result['error'])}"}), 500  # Return the error response

#         return jsonify({"message": "Facial recognition analysis started successfully", "result": result}), 200

#     except Exception as e:
#         return jsonify({"error": f"Facial recognition analysis failed: {str(e)}"}), 500

# @app.route('/get-speech-emotions/<id>', methods=['GET'])
# def getSpeechEmotions(id):
#     if not id:
#         return jsonify({"error": "Please provide id"}), 400
    
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
#     if error_message:
#         return jsonify({"error": error_message}), status_code
    
#     audio_url = interview_details["uploadDetails"]["audioUrl"]
#     if not audio_url:
#         return jsonify({"error": "Audio link not found in interview details"}), 404
#     try:
#         downloaded_file_name = download_file_from_s3(audio_url, "Audio")
#         print("DOWNLOADED PATH ", downloaded_file_name)
#         if downloaded_file_name:
#             print(f"Downloaded audio to: {downloaded_file_name}")
#         else :
#             raise Exception("Unable to download requested file")
#     except Exception as e:
#         return jsonify({"error": f"Failed to download audio: {str(e)}"}), 500

#     try:
#         update_stage(id, "speechEmotions", "started")  # Set stage to started

#         # Analyze facial emotions in a separate thread
#         audio_event.set()
#         result = analyze_speech_emotion_in_background(downloaded_file_name, id)
#         if 'error' in result:
#             update_stage(id, "speechEmotions", "failed")  # Set stage to failed
#             print(f"Error analyzing speech emotions: {result['error']}")
#             return jsonify({"error": f"Speech recognition analysis failed: {str(result['error'])}"}), 500  # Return the error response

#         return jsonify({"message": "Speech recognition analysis started successfully", "result": result}), 200

#     except Exception as e:
#         return jsonify({"error": f"Speech recognition analysis failed: {str(e)}"}), 500

# @app.route('/get-transcript/<id>', methods=['GET'])
# def getTranscript(id):
#     if not id:
#         return jsonify({"error": "Please provide id"}), 400
    
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
#     if error_message:
#         return jsonify({"error": error_message}), status_code
    
#     transcribe_event.set()
#     audio_url = interview_details["uploadDetails"]["audioUrl"]
#     if not audio_url:
#         return jsonify({"error": "Audio link not found in interview details"}), 404
#     try:
#         downloaded_file_name = download_file_from_s3(audio_url, "Audio")
#         print("DOWNLOADED PATH ", downloaded_file_name)
#         if downloaded_file_name:
#             print(f"Downloaded audio to: {downloaded_file_name}")
#         else :
#             raise Exception("Unable to download requested file")
#     except Exception as e:
#         return jsonify({"error": f"Failed to download audio: {str(e)}"}), 500
        
#     try:
#         # audio_filename = f"{id}.mp3"            
#         update_stage(id, "transcript", "started")
#         result = analyze_speech_to_text(downloaded_file_name, id)
#         return jsonify({"message": "Transcript analysis started successfully", "result": result}), 200
#     except Exception as e:
#         return jsonify({"error": f"Transcript analysis failed: {str(e)}"}), 500
    
# @app.route('/get-compare/<id>', methods=['GET'])
# def getCompare(id):
#     if not id:
#         return jsonify({"error": "Please provide id"}), 400
    
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
#     if error_message:
#         return jsonify({"error": error_message}), status_code
    
#     if interview_details['transcript']['data'] == None:
#         return jsonify({'error':'Transcript is not found, Run the "/get-transcript/<id>" api first'}), 404
    
#     compare_event.set()
#     transcript = interview_details['transcript']['data']
#     # with open(f'tmp/{id}.txt', 'w') as f:
#     #     f.write(transcript)
#     try:
#         # transcript_filename = f"{id}.txt"
#         update_stage(id, "comparePercentage", "started")
#         result = analyze_compare(transcript, id)
#         return jsonify({"message": "Compare analysis started successfully", "result": result}), 200
#     except Exception as e:
#         return jsonify({"error": f"Compare analysis failed: {str(e)}"}), 500
    
# @app.route('/interview/<id>', methods=['GET'])
# def get_interview(id):
#     interview_details, error_message, status_code = get_interview_details_by_id(id)
    
#     if error_message:
#         return jsonify({"error": error_message}), status_code
    
#     return { "message" : "successfully fetched interview details!", "data" : interview_details}, 200

# if __name__=="__main__":
#     app.run(port=8080,debug=True)