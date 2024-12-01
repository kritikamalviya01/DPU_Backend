# from flask import Flask, jsonify, request, send_file, stream_with_context, Response
# import os
# import threading
# from flask_sse import sse
# from src.speech_to_text import speech_to_text
# from src.facial_emotion import facial_emotion
# from src.speech_emotion1 import speechEmotionRecognition
# from utils.database.candidates import *
# from utils.extract_audio import extract_audio
# from utils.s3_storage import upload_to_s3

# app = Flask(__name__)
# app.config["REDIS_URL"] = "redis://localhost:6379"
# app.register_blueprint(sse, url_prefix='/stream')

# UPLOAD_FOLDER = 'recorded_video'
# if not os.path.exists(UPLOAD_FOLDER): 
#     os.makedirs(UPLOAD_FOLDER)

# def background_process(filename, file_path, objectId):
#     """Background task that performs all steps and streams progress."""
#     with app.app_context():  # Ensure Flask app context is available in the thread
#         def send_event(message):
#             sse.publish({"message": message}, type='progress')

#         # Step 1: Upload video to S3
#         send_event("Uploading video to S3...")
#         object_name = f"{filename[:-4]}_{str(objectId)}.mp4"
#         video_url = upload_to_s3(file_path, object_name)
#         update_into_db(objectId, name=filename[:-4], type="Video", title=object_name, url=video_url)
#         send_event("Video uploaded successfully!")

#         # Step 2: Extract Audio
#         send_event("Extracting audio from video...")
#         audio_file = extract_audio(file_path)
#         if isinstance(audio_file, dict) and "error" in audio_file:
#             send_event(f"Error in extracting audio: {audio_file['error']}")
#             return
        
#         send_event("Audio extracted successfully!")

#         # Step 3: Analyze Facial Emotion
#         send_event("Analyzing facial emotions...")
#         prediction = facial_emotion(filename)
#         update_into_db(objectId=objectId, facial_emotions=prediction)
#         send_event("Facial emotion analysis completed!")

#         # Step 4: Analyze Speech Emotion
#         send_event("Analyzing speech emotions...")
#         model = os.path.join('Models', 'audio.hdf5')
#         SER = speechEmotionRecognition(model)
#         emotions, timestamp = SER.predict_emotion_from_file(audio_file)
#         major_emotion = max(set(emotions), key=emotions.count)
#         emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
#         data = {
#             'emotions': emotions,
#             'major_emotion': major_emotion,
#             'emotion_dist': emotion_dist
#         }
#         update_into_db(objectId=objectId, speech_emotions=data)
#         send_event("Speech emotion analysis completed!")

#         # Step 5: Speech to Text
#         send_event("Transcribing speech to text...")
#         text, status_code = speech_to_text(audio_file, filename[:-4])
#         update_into_db(objectId=objectId, transcript=text)
#         send_event("Speech transcription completed!")

#         send_event("All processes completed successfully!")


# @app.route('/upload_and_process', methods=['POST'])
# def upload_and_process():
#     f = request.files.get('video')
#     audio = request.files.get("audio")

#     if not f and not audio:
#         return jsonify({"error": "No file provided"}), 400

#     filename = f.filename if f else audio.filename
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     if f:
#         f.save(file_path)
#     elif audio:
#         if not os.path.exists("recorded_audio"): 
#             os.makedirs("recorded_audio")
#         file_path = os.path.join("recorded_audio", filename)
#         audio.save(file_path)

#     objectId, null_document = create_null_document()

#     # Start background process
#     threading.Thread(target=background_process, args=(filename, file_path, objectId)).start()

#     return jsonify({"message": "File uploaded successfully. Processing started!", "filename": filename}), 200


# @app.route('/progress')
# def progress_stream():
#     """Endpoint to listen for progress updates."""
#     return Response(stream_with_context(sse), content_type='text/event-stream')


# if __name__ == "__main__":
#     app.run(port=8080, debug=True)




import threading
from flask import Flask, jsonify, request, render_template, send_file
import os
import subprocess
from flask_cors import CORS
import pandas as pd

from src.speech_to_text import speech_to_text
from src.facial_emotion import facial_emotion
from src.speech_emotion1 import speechEmotionRecognition
from src.compare import compare

from utils.database.candidates import *
from utils.database.pre_defined_questions import retrive_preDefinedQA
from utils.extract_audio import extract_audio
from utils.s3_storage import upload_to_s3

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'recorded_video'
if not os.path.exists(UPLOAD_FOLDER): 
    os.makedirs(UPLOAD_FOLDER)

RECORDED_AUDIO = 'recorded_audio'
if not os.path.exists(RECORDED_AUDIO): 
    os.makedirs(RECORDED_AUDIO)

audio_event = threading.Event()
transcript_event = threading.Event()

def process_video(file_path, objectId, filename):
    """Background task for processing video"""
    video_url = upload_to_s3(file_path, filename)
    update_into_db(objectId, name=filename[:-4], type="Video", title=filename, url=video_url)

    audio_file = extract_audio(file_path)
    audio_event.set()
    if isinstance(audio_file, dict) and "error" in audio_file:
        return jsonify({"error": audio_file["error"]})

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    audio = request.files.get("audio")
    if not video and not audio:
        return jsonify({"error": "No file provided"}), 400

    global filename
    global objectId

    if video:
        filename = video.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(file_path)

        objectId, null_document = create_null_document()
        # Run video processing in a separate thread

        process_thread = threading.Thread(target=process_video, args=(file_path, objectId, filename,))
        process_thread.start()
        # process_thread.join()
        update_fields = {}

        try:
            update_fields["facialEmotions.stage"] = "Processing"
            threading.Thread(target=analyze_facial_emotion_in_background, args=(filename,)).start()
        except:
            update_fields["facialEmotions.stage"] = "Failed"

        try:
            update_fields["speechEmotions.stage"] = "Processing"
            threading.Thread(target=analyze_speech_emotion_in_background, args=(filename,)).start()
        except:
            update_fields["speechEmotions.stage"] = "Failed"

        try:
            update_fields["transcript.stage"] = "Processing"
            threading.Thread(target=analyze_speech_to_text, args=(filename,)).start()
        except:
            update_fields["transcript.stage"] = "Failed"

        try:
            update_fields["comparePercentage.stage"] = "Processing"
            threading.Thread(target=analyze_compare, args=(filename,)).start()
        except:
            update_fields["comparePercentage.stage"] = "Failed"

        update_index(objectId, update_fields)

    elif audio:
        filename = audio.filename
        if not os.path.exists("recorded_audio"): 
            os.makedirs("recorded_audio")
        file_path = os.path.join("recorded_audio", filename)
        audio.save(file_path)
        objectId, null_document = create_null_document()
        object_name = f"{filename[:-4]}_{str(objectId)}.mp3"
        audio_url = upload_to_s3(file_path, object_name)
        update_into_db(objectId, name=filename[:-4], type="Audio", title=object_name, url=audio_url)

    return jsonify({"message": f"File saved successfully,", "filename": filename}), 200

def analyze_facial_emotion_in_background(filename):
    """Background task for analyzing facial emotions"""
    prediction = facial_emotion(filename)
    update_fields = {}
    update_fields["facialEmotions.stage"] = "Completed"
    update_index(objectId, update_fields)
    update_into_db(objectId=objectId, facial_emotions=prediction)

@app.route('/get_facial_emotions', methods=['GET'])
def get_facial_emotion():
    filename = request.args.get('filename')  # Get filename from request params
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        # Run facial emotion analysis in the background
        threading.Thread(target=analyze_facial_emotion_in_background, args=(filename,)).start()
        return jsonify({"message": "Facial emotion analysis started", "status_code": 200}), 200
    else:
        return jsonify({"error": "File not found", "status_code": 404}), 404

def analyze_speech_emotion_in_background(filename):
    """Background task for analyzing speech emotions"""
    audio_event.wait()
    file_path = os.path.join('recorded_audio', filename[:-1] + '3')
    model = os.path.join('Models', 'audio.hdf5')
    SER = speechEmotionRecognition(model)
    emotions, timestamp = SER.predict_emotion_from_file(file_path)

    major_emotion = max(set(emotions), key=emotions.count)
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    data = {
        'emotions': emotions,
        'major_emotion': major_emotion,
        'emotion_dist': emotion_dist
    }
    update_fields = {}
    update_fields["speechEmotions.stage"] = "Completed"
    update_index(objectId, update_fields)
    update_into_db(objectId=objectId, speech_emotions=data)

@app.route('/get_speech_emotion', methods=['GET'])
def get_speech_emotion():
    filename = request.args.get('filename')  # Get filename from request params
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    file_path = os.path.join('recorded_audio', filename[:-1] + '3')
    if os.path.exists(file_path):
        # Run speech emotion analysis in the background
        threading.Thread(target=analyze_speech_emotion_in_background, args=(filename,)).start()
        return jsonify({"message": "Speech emotion analysis started", "status_code": 200}), 200
    else:
        return jsonify({"error": "File not found", "status_code": 404}), 404
    
def analyze_speech_to_text(filename):
    audio_event.wait()
    file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
    if os.path.exists(file_path):
        text, status_code = speech_to_text(file_path, filename[:-4])  # Exclude extension
        update_fields = {}
        update_fields["transcript.stage"] = "Completed"
        update_index(objectId, update_fields)
        update_into_db(objectId=objectId, transcript=text)
    transcript_event.set()
    
@app.route('/speech_to_text', methods=['GET'])
def get_text():
    filename = request.args.get('filename')  # Get filename from request params
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
    if os.path.exists(file_path):
        threading.Thread(target=analyze_speech_to_text, args=(filename,)).start()
        return jsonify({"message": "Speech-to-text analysis started", "status_code": 200}), 200
    else:
        return jsonify({"error": "File not found", "status_code": 404}), 404
    
def analyze_compare(filename):
    transcript_event.wait()
    compare_score = compare(filename[:-4])  # Assuming `compare()` accepts a filename as a parameter
    update_fields = {}
    update_fields["comparePercentage.stage"] = "Completed"
    update_index(objectId, update_fields)
    update_into_db(objectId=objectId, compare_percentage=float(compare_score))

@app.route('/compare', methods=['GET', 'POST'])
def compare_text():
    filename = request.args.get('filename')  # Get filename from request params
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    # Perform comparison logic
    threading.Thread(target=analyze_compare, args=(filename,)).start()
    return jsonify({"message": "Comparision analysis started", "status_code": 200}), 200


if __name__ == "__main__":
    app.run(port=8080, debug=True)
