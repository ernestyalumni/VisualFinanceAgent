from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda','trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.schema import Document
import os
import json

def return_index(path):
    docs = []
    for dir in os.listdir(path):
        pdfs = os.path.join(path,dir)
        for json_path in os.listdir(os.path.join(pdfs,"JSON")):
            with open(os.path.join(pdfs,"JSON",json_path), 'r') as file:
                data = json.load(file)
            docs.append(Document(page_content=data['summary'],metadata={"filename":dir,"page_num":json_path}))

    db = FAISS.from_documents(docs, hf)

    return db.as_retriever()