import dspy
from byaldi import RAGMultiModalModel
import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from query_utils import query_translator, query_enrichment
import collections

QueryImgTuple = collections.namedtuple("QueryImgTuple",['query','image_base64'])

class PipeLine(dspy.Module):
    def __init__(self):
        self.vision_index = self._get_vision_index()
        self.summary_index = _get_summary_index("visualfinanceagent/vectordb/output_imgs_2")
    
    def _get_vision_index(self):
        INDEX_NAME = "finance_data"
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
        search_index = RAG.from_index(INDEX_NAME)
        return search_index
    
    def __call__(self,user_query:str):
        translated_query = query_translator(user_query)
        
        relevant_summaries = self.summary_index.invoke(translated_query)
        
        summaries = ""
        for rs in relevant_summaries:
            summaries+=rs.page_content + "\n"
        
        enriched_queries = query_enrichment(user_query, translated_query, summaries)
        
        relevant_img_results = []
        
        for eq in enriched_queries:
            relevant_imgs = self.vision_index.search(query=eq,k=3)
            relevant_img_results.append(
                QueryImgTuple(query=eq,image_base64=[i['base64'] for i in relevant_imgs])
            )

        return translated_query, summaries, relevant_img_results
    
def _get_summary_index(path):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda','trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    docs = []
    for dir in os.listdir(path):
        pdfs = os.path.join(path,dir)
        for json_path in os.listdir(os.path.join(pdfs,"JSON")):
            with open(os.path.join(pdfs,"JSON",json_path), 'r') as file:
                data = json.load(file)
            docs.append(Document(page_content=data['summary'],metadata={"filename":dir,"page_num":json_path}))

    db = FAISS.from_documents(docs, hf)

    return db.as_retriever()      