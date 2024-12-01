import os
from pymongo import MongoClient
import boto3
from botocore.exceptions import NoCredentialsError
from flask import Flask, request, jsonify
# from dotenv import load_dotenv

# load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client['ai_interview']
collection = db['video_info']

# S3 setup
s3_client = boto3.client('s3',
                         aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
                         aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"))
BUCKET_NAME = os.getenv("BUCKET_NAME")

def upload_to_s3(file_name, object_name):
    try:
        response = s3_client.upload_file(file_name, BUCKET_NAME, object_name)
        print(response)
        return f'https://{BUCKET_NAME}.s3.amazonaws.com/{object_name}'
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

def upload_video():
    file = request.files['video']
    candidate_name = request.form['name']
    file_name = f"{candidate_name}_{file.filename}"
    
    file.save(file_name)
    video_url = upload_to_s3(file_name, file_name)
    
    if video_url:
        document = {
            'name': candidate_name,
            'video_url': video_url,
            'other_details': request.form.get('other_details', ''),
        }
        collection.insert_one(document)
        os.remove(file_name)  # Clean up local file
        return ({'message': 'Video uploaded successfully!', 'video_url': video_url})
    else:
        return ({'message': 'Video upload failed!'})

def download_file_from_s3(fileUrl, type):

    fileName = fileUrl.split('/')[-1]

    print("DOWNLOAD FILE NAME ---->",fileName)
    try:
        if (type == "Audio"):
            tempPath =  f"recorded_audio/{fileName}"
            s3_client.download_file(BUCKET_NAME, fileName, tempPath )
            if os.path.exists(tempPath):
                return fileName
            else:
                raise Exception("Unable to download file")
        
        if(type == 'Video'):
            tempPath =  f"recorded_video/{fileName}"
            s3_client.download_file(BUCKET_NAME, fileName, tempPath )
            if os.path.exists(tempPath):
                return fileName
            else:
                raise Exception("Unable to download file")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None
