import dspy
from byaldi import RAGMultiModalModel
import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import collections
from groq import AsyncGroq

QueryImgTuple = collections.namedtuple("QueryImgTuple",['query','image_base64'])

class PipeLine(dspy.Module):
    def __init__(self):
        self.vision_index = self._get_vision_index()
        self.summary_index = _get_summary_index("visualfinanceagent/vectordb/output_imgs_2")
        self.groq_client = AsyncGroq(api_key=os.environ['GROQ_API_KEY'])

    def _get_vision_index(self):
        INDEX_NAME = "finance_data"
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
        search_index = RAG.from_index(INDEX_NAME)
        return search_index
    
    async def groq_response(self,image_base64, question):
        completion = await self.groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        # response_format={"type": "json_object"},
        stop=None,
        )
        # return SummaryResponse.model_validate_json(chat_completion.choices[0].message.content)
        return completion.choices[0].message.content
    
    async def __call__(self,user_query:str):
        #Translate the simple user query to better query
        query_translator = self.query_translator(user_query)
        
        relevant_summaries = self.summary_index.invoke(query_translator)
        
        summaries = ""
        for rs in relevant_summaries:
            summaries+=rs.page_content + "\n"
        
        #Based on relevant summaries, it translates the user query into three enriched queries
        enriched_queries = self.query_enrichment(user_query, query_translator, summaries)
        
        relevant_img_results: list[QueryImgTuple] = []
        
        for eq in enriched_queries:
            relevant_imgs = self.vision_index.search(query=eq,k=3)
            relevant_img_results.append(
                QueryImgTuple(query=eq,image_base64=[(i['base64'], await self.groq_response(i['base64'],eq)) for i in relevant_imgs])
            )

        manager_response_list:list[str] = []
        for ri in relevant_img_results:
            for r in ri.image_base64:
                manager_response_list.append(
                    r[1]
                )
        #Manager response
        manager_response = self.manager_response(manager_response_list,query_translator, user_query)
        return query_translator, summaries, relevant_img_results, manager_response
    
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