import os
import subprocess

# def extract_audio(video_path):
#     # Create the audio filename in the recorded_audio directory
#     audio_filename = os.path.splitext(os.path.basename(video_path))[0] + '.mp3'
#     audio_path = os.path.join('recorded_audio', audio_filename)
    
#     if os.path.exists(audio_path):
#         os.remove(audio_path)

#     command = [
#         'ffmpeg', '-i', video_path,
#         '-q:a', '0', '-map', 'a', audio_path
#     ]
#     try:
#         subprocess.run(command, check=True)
#         return audio_path
#     except subprocess.CalledProcessError as e:
#         return {"error": f"Audio extraction failed: {str(e)}"}

def extract_audio(video_path):
    # Create the audio filename in the recorded_audio directory
    audio_filename = os.path.splitext(os.path.basename(video_path))[0] + '.mp3'
    audio_path = os.path.join('recorded_audio', audio_filename)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    # Remove the file if it already exists
    if os.path.exists(audio_path):
        os.remove(audio_path)

    command = [
        'ffmpeg', '-i', video_path,
        '-q:a', '0', '-map', 'a', audio_path
    ]
    try:
        subprocess.run(command, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        # print("error audio ext ", str(e))
        return {"error": f"Audio extraction failed: {str(e)}"}
    except Exception as e:
        # print("error exc ", str(e))
        return {"error": f"An unexpected error occurred: {str(e)}"}
