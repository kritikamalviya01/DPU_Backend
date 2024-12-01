import pymongo
from bson.objectid import ObjectId
import os

myclient = pymongo.MongoClient(os.getenv("MONGODB_URI"))

mydb = myclient.ai_interview
collections = mydb.candidates

def create_null_document(type:str, create=True):
    objectId = None
    null_document = {
        "candidateName" : None,
        "type" : type,
        "uploadDetails" : {
            "title" : None,
            "videoUrl" : None,
            "audioUrl" : None
        },
        "facialEmotions" :{
            "stage" : "started",
            "data" : None
        },
        "speechEmotions" : {
            "stage" : "started",
            "data" : None
        },
        "transcript" : {
            "stage" : "started",
            "data" : None
        },
        "comparePercentage" : {
            "stage" : "started",
            "data" : None
        },
    }
    if type == "Audio":
        del null_document["facialEmotions"]
        del null_document["uploadDetails"]["videoUrl"]
    if create:
        objectId = collections.insert_one(null_document).inserted_id
        print(f"Null Document Created with {objectId}")
    return objectId, null_document

def save_into_db(name, url, facial_emoitions, speech_emotions, transcript, compare_percentage):
    candidate = {
        "candidate_name" : name,
        "videoDetails" : {
            "title" : name+".mp4",
            "url" : url
        },
        "facial_emoitions" :facial_emoitions,
        "speech_emotions" : speech_emotions,
        "transcript" : transcript,
        "compare_percentage" : float(compare_percentage)
    }

    collections.insert_one(candidate)
    return "Predictions Saved Successfully"

# def update_index(objectId, update_fields):
#     result = collections.update_one(
#         {'_id' : objectId},
#         {'$set': update_fields}
#     )
#     if result.modified_count > 0:
#         print("Document updated successfully")
#     else:
#         print("No document was updated")
#     return result

def update_index(objectId, update_fields):
    # Use find_one_and_update to update the document and return the updated document
    # print("update fields ---------> ", update_fields)
    updated_document = collections.find_one_and_update(
        {'_id': ObjectId(objectId)},
        {'$set': update_fields},
        return_document=True  # Return the updated document
    )

    if updated_document:
        updated_document['_id'] = str(updated_document['_id'])
        print("Document updated successfully")
    else:
        print("No document was updated or document not found")
    
    return updated_document



# def update_into_db(objectId, name=None, title=None, url=None, facial_emotions=None, speech_emotions=None, transcript=None, compare_percentage=None):
#     update_fields = {
#         "candidate_name" : name,
#         "videoDetails" : {"title" : title}
#     }
#     if name:
    
#         update_index(objectId,update_fields)

#     if url:
#         update_fields ={
#             "videoDetails" : {"url" : url}
#         }
#         #update_index(objectId,update_fields)
    
#     if facial_emotions:
#         update_fields = {
#             "facial_emoitions" : facial_emotions
#         }
#         update_index(objectId,update_fields)
#     return "updated into Database"

def updateFieldsIntoDB(objectId, update_fields):
    result = update_index(objectId, update_fields)

    return result

def update_into_db(objectId, null_document=None, name:str=None, type:str=None, title:str=None, 
                   video_url:str=None, audio_url:str=None, facial_emotions:object=None, speech_emotions:object=None, 
                   transcript:str=None, compare_percentage:float=None):
    # Create a deep copy of the null_document to prevent modifying the original
    # _, update_fields = create_null_document(create=False)
    update_fields = {}

    # Update fields with provided values or keep them as None
    if name is not None:
        update_fields["candidateName"] = name

    if type is not None:
        update_fields["type"] = type
    
    if title is not None or video_url is not None or audio_url is not None:
        videoDetails = {}
        if title is not None:
            videoDetails["title"] = title
        if video_url is not None:
            videoDetails["videoUrl"] = video_url
        if audio_url is not None:
            videoDetails["audioUrl"] = audio_url
        update_fields["uploadDetails"] = videoDetails
    
    if facial_emotions is not None:
        update_fields["facialEmotions.data"] = facial_emotions
    
    if speech_emotions is not None:
        update_fields["speechEmotions.data"] = speech_emotions
    
    if transcript is not None:
        update_fields["transcript.data"] = transcript
    
    if compare_percentage is not None:
        update_fields["comparePercentage.data"] = compare_percentage

    # Call update_index with the fully populated update_fields
    result = update_index(objectId, update_fields)

    return result

def save_video_metadata_into_db(name,video_url,other_details):
    video_metadata = {
        "name" : name,
        "video_url" : video_url,
        "other_details" : other_details
    }
    collections.insert_one(video_metadata)
    return "Video MetaData Saved Successfully"


def get_interview_details_by_id(interview_id):
    try:
        # Validate ObjectId
        if not ObjectId.is_valid(interview_id):
            return None, "Invalid ID format", 400
        
        # Fetch interview details
        interview_details = collections.find_one({"_id": ObjectId(interview_id)})
        
        if interview_details is None:
            return None, "Interview not found", 404

        # Convert _id to string
        interview_details['_id'] = str(interview_details['_id'])
        
        return interview_details, None, 200
    
    except Exception as e:
        return None, str(e), 500

def retrive_data(name):
    data = collections.find({},{"candidate_name" : name})
    return data

