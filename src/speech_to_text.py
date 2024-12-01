import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Ensure the directory for output text exists
if not os.path.exists('tmp'):
    os.makedirs('tmp')

def speech_to_text(audio_file_path):
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model ID
    model_id = "distil-whisper/distil-large-v3"

    # Load model and processor
    try:
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
    except Exception as e:
        return None, f"Failed to load model or processor: {str(e)}"

    # Transcribe audio
    try:
        result = pipe(audio_file_path, return_timestamps=True)  # Enable timestamps for long audio
        return result["text"], None
    except Exception as e:
        return None, f"Failed to transcribe audio: {str(e)}"


# def speech_to_text(audio_file_path):
#     # Set device and dtype
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     # Model ID
#     model_id = "distil-whisper/distil-large-v3"

#     # Load model and processor
#     try:
#         model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#         )
#         model.to(device)

#         processor = AutoProcessor.from_pretrained(model_id)

#         # Create the speech recognition pipeline
#         pipe = pipeline(
#             "automatic-speech-recognition",
#             model=model,
#             tokenizer=processor.tokenizer,
#             feature_extractor=processor.feature_extractor,
#             max_new_tokens=128,
#             torch_dtype=torch_dtype,
#             device=device,
#         )
#     except Exception as e:
#         return {"error": f"Failed to load model or processor: {str(e)}", "status_code": 500}

#     # Transcribe audio
#     try:
#         result = pipe(audio_file_path, return_timestamps=True)  # Enable timestamps for long audio
#         return result["text"]
#     except Exception as e:
#         return {"error": f"Failed to transcribe audio: {str(e)}", "status_code": 500}

# def speech_to_text(audio_file_path, filename):
#     # Set device and dtype
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     # Model ID
#     model_id = "distil-whisper/distil-large-v3"

#     # Load model and processor
#     model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#     )
#     model.to(device)

#     processor = AutoProcessor.from_pretrained(model_id)

#     # Create the speech recognition pipeline
#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         max_new_tokens=128,
#         torch_dtype=torch_dtype,
#         device=device,
#     )

#     # Transcribe audio
#     try:
#         result = pipe(audio_file_path, return_timestamps=True)  # Enable timestamps for long audio
#     except Exception as e:
#         return {"error": f"Failed to transcribe audio: {str(e)}", "status_code": 500}

#     # Write the result to a text file
#     with open(f"tmp/{filename}.txt", "w") as text_file:
#         text_file.write(result["text"])

#     return result["text"], 200

# Example usage:
# text = speech_to_text("path/to/your/audiofile.wav")
# print(text)
