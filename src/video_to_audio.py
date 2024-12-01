from moviepy.editor import VideoFileClip
import tempfile
from io import BytesIO
import os

def convert_video_to_audio(video_data, filename):
    # Save the BytesIO content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(video_data.read())  # Write the BytesIO content to the temporary file
        temp_file.flush()  # Ensure all data is written

        # Use moviepy to load the video and extract the audio
        video_clip = VideoFileClip(temp_file.name)

        # Extract the audio
        audio_clip = video_clip.audio

        # Save the audio to a file (e.g., mp3 or wav)
        audio_output_path = os.path.join('recorded_video',filename)
        audio_clip.write_audiofile(audio_output_path)

        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()

    return f'Audio extracted and saved to {audio_output_path}'
