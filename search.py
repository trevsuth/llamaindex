import pymongo
import os
from dotenv import load_dotenv
from utils.llm import embed, chat

load_dotenv()

text = 'How do healing potions work?'
vectored_text = embed(text)

#mongo vars
pipeline = [
    {
        '$vectorSearch': {
            'index': 'vector_index',
            'path': 'embedding',
            'queryVector': vectored_text,
            'numCandidates': 150,
            'limit': 3
        }
    }, {
        '$project': {
            '_id': 0,
            'text' : 1,
            'score': {
                '$meta': 'vectorSearchScore'
            }
        }
    }
]

mongo_uri = os.getenv('MONGO_URI')
db_name=os.getenv('DB_NAME')
collection_name=os.getenv('COLLECTION_NAME')
index_name=os.getenv('INDEX_NAME')

# remove docs from collection
mongodb_client = pymongo.MongoClient(mongo_uri)

db = mongodb_client[db_name]
collection = db[collection_name]

result = collection.aggregate(pipeline=pipeline)

for i in result:
    print(i)
    
msg = 'Summarize the following in a single paragraph'    

chat()