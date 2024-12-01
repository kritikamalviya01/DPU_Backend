import threading
from flask import Flask, jsonify, request, render_template
import os
import requests
from flask_cors import CORS

from src.speech_to_text import speech_to_text
from src.facial_emotion import facial_emotion
from src.speech_emotion1 import speechEmotionRecognition
from src.compare import compare

from utils.database.pre_defined_questions import retrive_preDefinedQA
from utils.database.candidates import *
from utils.extract_audio import extract_audio
from utils.s3_storage import upload_to_s3, download_file_from_s3

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'recorded_video'
AUDIO_FOLDER = 'recorded_audio'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

def process_video(file_path, objectId, filename):
    """Background task for processing video."""
    print('"""Background task for processing video."""')
    try:
        print('file path', file_path)
        audio_file = extract_audio(file_path)
        if isinstance(audio_file, dict) and "error" in audio_file:
            print("Error in audio extraction:", audio_file["error"])
            raise Exception(audio_file["error"])

        
        # Ensure audio file exists before proceeding
        audio_file_path = audio_file # Adjust this based on your extract_audio return structure
        if not os.path.exists(audio_file_path):
            print(f"Audio file not found: {audio_file_path}")
            raise FileNotFoundError(f"Audio file could not be found: {audio_file_path}")

        print("audio file path ", audio_file_path)

        # Initiate further analyses in separate threads
        threading.Thread(target=analyze_facial_emotion_in_background, args=(filename, objectId)).start()
        threading.Thread(target=analyze_speech_emotion_in_background, args=(audio_file_path, objectId)).start()
        threading.Thread(target=analyze_speech_to_text, args=(audio_file_path, objectId)).start()
        # threading.Thread(target=analyze_compare, args=(filename, objectId)).start()

    except Exception as e:
        print(f"Error processing video: {str(e)}")

def process_audio(file_path, objectId, filename):
    """Background task for processing video."""
    print('"""Background task for processing video."""')
    try:
        print('file path', file_path)
       
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            raise FileNotFoundError(f"Audio file could not be found: {file_path}")

        threading.Thread(target=analyze_speech_emotion_in_background, args=(file_path, objectId)).start()
        threading.Thread(target=analyze_speech_to_text, args=(file_path, objectId)).start()

    except Exception as e:
        print(f"Error processing video: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload for video or audio."""
    # Retrieve uploaded video and audio files
    uploaded_video = request.files.get('video')
    uploaded_audio = request.files.get('audio')

    # Check if neither file was provided
    if not uploaded_video and not uploaded_audio:
        return jsonify({"error": "No file provided"}), 400

    # Handle video file upload
    if uploaded_video:
        filename = uploaded_video.filename
        if not filename:
            return jsonify({"error": "No filename provided for video."}), 400
        print('filename ', filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print('Video file path:', file_path)

        try:
            uploaded_video.save(file_path)
            objectId, _ = create_null_document()
            video_url = upload_to_s3(file_path, filename)
            result = update_into_db(objectId, name=filename[:-4], type="Video", title=filename, url=video_url)
            threading.Thread(target=process_video, args=(file_path, objectId, filename)).start()

            return jsonify({"message": "Video uploaded successfully.", "filename": filename, "data" : result}), 200

        except Exception as e:
            return jsonify({"error": f"Failed to upload video: {str(e)}"}), 500

    # Handle audio file upload
    if uploaded_audio:
        filename = uploaded_audio.filename
        if not filename:
            return jsonify({"error": "No filename provided for audio."}), 400
        
        file_path = os.path.join(AUDIO_FOLDER, filename)
        print('Audio file path:', file_path)

        try:
            uploaded_audio.save(file_path)
            objectId, _ = create_null_document()
            audio_url = upload_to_s3(file_path, filename)
            result = update_into_db(objectId, name=filename[:-4], type="Audio", title=filename, url=audio_url)
            threading.Thread(target=process_audio, args=(file_path, objectId, filename)).start()

            return jsonify({"message": "Audio uploaded successfully.", "filename": filename, "data" : result}), 200

        except Exception as e:
            return jsonify({"error": f"Failed to upload audio: {str(e)}"}), 500

def update_stage(objectId, field, stage, data=None):
    """Update the stage and optional data for a specific field in the database."""
    update_fields = {f"{field}.stage": stage}
    if data is not None:
        update_fields[f"{field}.data"] = data
    result = updateFieldsIntoDB(objectId, update_fields)
    return result

def analyze_facial_emotion_in_background(filename, objectId):
    """Background task for analyzing facial emotions."""
    print("""Background task for analyzing facial emotions.""")
    update_stage(objectId, "facialEmotions", "started")  # Set stage to started
    try:
        prediction = facial_emotion(filename)
        # Check if the prediction contains an error
        if 'error' in prediction:
            # update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
            # print(f"Error analyzing facial emotions: {prediction['error']}")
            # return   # Return the error response
            raise Exception("Error analyzing facial emotions: {prediction['error']}")

        print("facial emotion success")
        result = update_stage(objectId, "facialEmotions", "success", prediction)  # Set stage to success
        return result
    except Exception as e:
        update_stage(objectId, "facialEmotions", "failed")  # Set stage to failed
        print(f"Error analyzing facial emotions: {str(e)}")
    finally:
        # Remove the file after processing
        file_path = os.path.join('recorded_video', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed file: {filename}")
        else:
            print(f"File not found: {filename}")

def analyze_speech_emotion_in_background(audio_file, objectId):
    """Background task for analyzing speech emotions."""
    print("""Background task for analyzing speech emotions.""")
    update_stage(objectId, "speechEmotions", "started")  # Set stage to started
    try:
        model = os.path.join('Models', 'audio.hdf5')
        SER = speechEmotionRecognition(model)
        emotions, _ = SER.predict_emotion_from_file(audio_file)

        major_emotion = max(set(emotions), key=emotions.count)
        emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]
        
        data = {
            'emotions': emotions,
            'major_emotion': major_emotion,
            'emotion_dist': emotion_dist
        }
        print("success speech emotion")
        update_stage(objectId, "speechEmotions", "success", data)  # Set stage to success

    except Exception as e:
        update_stage(objectId, "speechEmotions", "failed")  # Set stage to failed
        print(f"Error analyzing speech emotions: {str(e)}")

def analyze_speech_to_text(audio_file, objectId):
    """Convert speech to text."""
    print("Starting speech-to-text conversion.")
    update_stage(objectId, "transcript", "started")  # Set stage to started

    try:
        text, error = speech_to_text(audio_file)  # Get transcription result
        if error:  # Check if there's an error message
            raise Exception(error)

        print("Transcription successful. Transcribed text:", text)
        update_stage(objectId, "transcript", "success", text)  # Set stage to success
        
        # Start the comparison in a separate thread
        threading.Thread(target=analyze_compare, args=(audio_file, objectId, text)).start()

    except Exception as e:
        update_stage(objectId, "transcript", "failed")  # Set stage to failed
        update_stage(objectId, "comparePercentage", "failed")  # Set compare percentage stage to failed
        print(f"Error during speech-to-text conversion: {str(e)}")



# def analyze_compare(filename, objectId):
#     """Perform comparison analysis."""
#     update_stage(objectId, "comparePercentage", "started")  # Set stage to started
#     try:
#         compare_score = compare(filename[:-4])  # Assuming compare() accepts a filename
#         update_stage(objectId, "comparePercentage", "success", float(compare_score))  # Set stage to success
#     except Exception as e:
#         update_stage(objectId, "comparePercentage", "failed")  # Set stage to failed
#         print(f"Error during comparison analysis: {str(e)}")

def analyze_compare(file_path, objectId, text):
    """Analyze the comparison of speech text with predefined questions."""
    # try:
    #     # Retrieve the speech text from the database
    #     # file_path = os.path.join('recorded_audio', filename[:-1] + '3')  # Adjusted to use filename directly
    #     speech_text, _ = speech_to_text(file_path)  # Get the transcribed text
    # except Exception as e:
    #     print(f"Error during speech-to-text conversion: {str(e)}")
    #     update_stage(objectId, "comparePercentage", "failed")
    #     return

    try:
        # Define parameters for retrieving questions
        level = "beginner"  # You can modify this as needed
        tech = "nodejs"  # You can modify this as needed
        answer = 1  # Example answer, adjust as necessary

        # Retrieve predefined questions
        documents = retrive_preDefinedQA(level, tech, answer)
        
        if not documents:
            print("No documents retrieved for comparison.")
            update_stage(objectId, "comparePercentage", "failed")
            return
        
        # Combine answers from retrieved questions
        combined_answers = ' '.join([doc['answer'] for doc in documents if 'answer' in doc])
    except Exception as e:
        print(f"Error during question retrieval: {str(e)}")
        update_stage(objectId, "comparePercentage", "failed")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        return

    try:
        # Perform comparison
        compare_score = compare(text, combined_answers)  # Assuming compare function can take both texts
        update_stage(objectId, "comparePercentage", "success", float(compare_score))  # Set stage to success
        # update_into_db(objectId=objectId, compare_percentage=float(compare_score))
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        update_stage(objectId, "comparePercentage", "failed")
    finally:
        # Remove the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        else:
            print(f"File not found: {file_path}")


@app.route('/retrieve', methods=['GET'])
def retrieve_questions():
    level = request.args.get('level')
    tech = request.args.get('tech')
    answer = request.args.get("answer")
    documents = retrive_preDefinedQA(level, tech, int(answer))
    return jsonify(documents)

@app.route('/get-facial-emotions/<id>', methods=['GET'])
def getFacialEmotions(id):
    if not id:
        return jsonify({"error": "Please provide id"}), 400
    
    # Step 1: Get interview details
    interview_details, error_message, status_code = get_interview_details_by_id(id)
    if error_message:
        return jsonify({"error": error_message}), status_code

    if interview_details['type'] == 'Audio':
         return jsonify({"error": "This is an audio interview cannot process video"}), 400

    # Step 2: Extract video URL from interview details
    video_url = interview_details["VideoDetails"]["url"]  # Adjust based on your data structure
    if not video_url:
        return jsonify({"error": "Video link not found in interview details"}), 404

    # Step 3: Download the video
    try:
        # response = requests.get(video_url)
        downloaded_file_name = download_file_from_s3(video_url, "Video")
        print("DOWNLOADED PATH ", downloaded_file_name)
        if downloaded_file_name:
        # # response.raise_for_status()  # Raise an error for bad responses
        #     # video_file_path = os.path.join(UPLOAD_FOLDER, f"{id}.mp4")  # Save with the interview ID
        #     with open(video_file_path, 'wb') as f:
        #         f.write(response.content)
            print(f"Downloaded video to: {downloaded_file_name}")
        else :
            raise Exception("Unable to download requested file")

    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 500

    # Step 4: Perform facial recognition analysis
    try:
        update_stage(id, "facialEmotions", "started")  # Set stage to started

        # Analyze facial emotions in a separate thread
        result = analyze_facial_emotion_in_background(downloaded_file_name, id)
        if 'error' in result:
            update_stage(id, "facialEmotions", "failed")  # Set stage to failed
            print(f"Error analyzing facial emotions: {result['error']}")
            return jsonify({"error": f"Facial recognition analysis failed: {str(result['error'])}"}), 500  # Return the error response

        return jsonify({"message": "Facial recognition analysis started successfully", "result": result}), 200

    except Exception as e:
        return jsonify({"error": f"Facial recognition analysis failed: {str(e)}"}), 500


    
    


@app.route('/interview/<id>', methods=['GET'])
def get_interview(id):
    interview_details, error_message, status_code = get_interview_details_by_id(id)
    
    if error_message:
        return jsonify({"error": error_message}), status_code
    
    return { "message" : "successfully fetched interview details!", "data" : interview_details}, 200

if __name__ == "__main__":
    app.run(port=8080, debug=True)
