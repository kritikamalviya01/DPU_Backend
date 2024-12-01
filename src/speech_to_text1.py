# import speech_recognition as sr

# r = sr.Recognizer()

# filename = "test.wma"

# # open the file
# with sr.AudioFile(filename) as source:
#     # listen for the data (load audio to memory)
#     audio_data = r.record(source)
#     # recognize (convert from speech to text)
#     text = r.recognize_google(audio_data)
#     print(text)

# # importing libraries 
# import speech_recognition as sr 
# import os 
# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# # create a speech recognition object
# r = sr.Recognizer()

# # a function to recognize speech in the audio file
# # so that we don't repeat ourselves in in other functions
# def transcribe_audio(path):
#     # use the audio file as the audio source
#     with sr.AudioFile(path) as source:
#         audio_listened = r.record(source)
#         # try converting it to text
#         text = r.recognize_google(audio_listened)
#     return text

# # a function that splits the audio file into chunks on silence
# # and applies speech recognition
# def get_large_audio_transcription_on_silence(path):
#     """Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks"""
#     # open the audio file using pydub
#     sound = AudioSegment.from_file(path)  
#     # split audio sound where silence is 500 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#         # experiment with this value for your target audio file
#         min_silence_len = 500,
#         # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
#         # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk 
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         # recognize the chunk
#         try:
#             text = transcribe_audio(chunk_filename)
#         except sr.UnknownValueError as e:
#             print("Error:", str(e))
#         else:
#             text = f"{text.capitalize()}. "
#             print(chunk_filename, ":", text)
#             whole_text += text
#     # return the text for all chunks detected
#     return whole_text

# path = "harvard.wav"
# print("\nFull text:", get_large_audio_transcription_on_silence(path))

# #import library
# import speech_recognition as sr
# #Initiаlize  reсоgnizer  сlаss  (fоr  reсоgnizing  the  sрeeсh)
# r = sr.Recognizer()
# # Reading Audio file as source
# #  listening  the  аudiо  file  аnd  stоre  in  аudiо_text  vаriаble
# with sr.AudioFile('harvard.wav') as source:
#     audio_text = r.listen(source)
# # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
#     try:
#         # using google speech recognition
#         text = r.recognize_google(audio_text)
#         print('Converting audio transcripts into text ...')
#         print(text)
#     except:
#          print('Sorry.. run again...')

import torch
import os
import moviepy.editor as mp
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Ensure the directory for recorded audio exists
if not os.path.exists('recorded_audio'): 
    os.makedirs('recorded_audio')

def speech_to_text(filename):
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model ID
    model_id = "distil-whisper/distil-large-v3"

    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load video file and extract audio
    video_path = os.path.join('recorded_video', filename)
    audio_path = os.path.join('recorded_audio', filename[:-4] + '.mp3')  # Change to .mp3

    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
    except Exception as e:
        return {"error": f"Failed to extract audio: {str(e)}"}

    # Transcribe audio
    try:
        result = pipe(audio_path)
    except Exception as e:
        return {"error": f"Failed to transcribe audio: {str(e)}"}

    # Write the result to a text file
    with open("tmp/transcribe.txt", "w") as text_file:
        text_file.write(result["text"])

    return result["text"]

#print(result["text"])


# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# #from datasets import load_dataset


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16

# model_id = "openai/whisper-large-v3"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# #dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# #sample = dataset[0]["audio"]

# result = pipe('voice_recording.wav')
# print(result["text"])
