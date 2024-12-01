import pymongo
from bson.objectid import ObjectId
import os

myclient = pymongo.MongoClient(os.getenv("MONGODB_URI"))

mydb = myclient.ai_interview
collections = mydb.pre_defined_answers

def retrive_preDefinedQA(level:str=None,tech:str=None, answer=1):
    query = {}
    if level:
        query["level"] = level
    if tech:
        query["tech"] = tech
    result = list(collections.find(query,{"_id" : 0, "answer" : answer}))
    return result



    