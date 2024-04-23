import pymongo
import os
from dotenv import load_dotenv

# llama index imports
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# for local llm and embeddinga
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# for chunking
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

embedding_model = os.getenv('EMBEDDING_MODEL')
llm_model = os.getenv('LLM_MODEL')

# index settings
documents = SimpleDirectoryReader("data").load_data()
embed_model = OllamaEmbedding(model_name=embedding_model)
llm = Ollama(model=llm_model, request_timeout=(120.0*5))

Settings.embed_model = embed_model
Settings.llm = llm

# create parser
parser = SentenceSplitter(chunk_size=100, chunk_overlap=10)
nodes = parser.get_nodes_from_documents(documents)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

#mongo vars
mongo_uri = os.getenv('MONGO_URI')
db_name=os.getenv('DB_NAME')
collection_name=os.getenv('COLLECTION_NAME')
index_name=os.getenv('INDEX_NAME')

# remove docs from collection
mongodb_client = pymongo.MongoClient(mongo_uri)

db = mongodb_client[db_name]
collection = db[collection_name]
collection.delete_many({})
                
store = MongoDBAtlasVectorSearch(mongodb_client,
                                 db_name=db_name,
                                 collection_name=collection_name,
                                 index_name=index_name
                                 )
store.add(nodes)

storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)