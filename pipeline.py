import dspy
from byaldi import RAGMultiModalModel
import os
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import collections
from groq import AsyncGroq
import pydantic
from typing import Annotated

QueryImgTuple = collections.namedtuple("QueryImgTuple",['query','image_base64'])

class EnrichedQuery(pydantic.BaseModel):
    enriched_queries: Annotated[str,"List of enriched queries separated "]
class VisionFinancePipeLine(dspy.Module):
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
    
    async def query_translator(self,user_query):
        completion = await self.groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in clarifying and expanding investment and consulting queries. Your task is to take a brief user query and generate a well-formed, detailed sentence that provides more context and depth. Focus on creating a full sentence with proper grammar that explores the main aspect of the original query. Return only the expanded query, nothing else."
            },
            {
                "role": "user",
                "content": f"Please expand the following query into a detailed sentence: '{user_query}'"
            }
        ],
        model="llama-3.1-70b-versatile",
    )
        return completion.choices[0].message.content
    
    async def query_enrichment(self, user_query, query_translator, summaries):
        
        chat_completion = await self.groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in generating enriched queries based on original queries, translated queries, and relevant summaries. Your task is to generate 3 enriched queries that explore different aspects of the topic. Return only the list of 3 enriched queries, separated by newlines."
            },
            {
                "role": "user",
                "content": f"Original query: '{user_query}'\nTranslated query: '{query_translator}'\nRelevant summaries: {summaries}\n\nPlease generate 3 enriched queries based on this information."
            }
        ],
        model="llama-3.1-70b-versatile",
    )
        return chat_completion.choices[0].message.content
    
    async def manager_response(self, manager_response_list, query_translator, user_query):
        summaries_with_ids = [f"Summary {i}: {summary}" for i, summary in enumerate(manager_response_list)]
        summaries_text = "\n\n".join(summaries_with_ids)
        
        chat_completion = await self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in evaluating the relevance of information to user queries. Your task is to analyze a list of summaries and determine which ones are most relevant to the user's original query and the translated query. Return only the IDs of the most relevant summaries as JSON object {json.dumps(EnrichedQuery.model_json_schema())}, separated by commas."
                },
                {
                    "role": "user",
                    "content": f"User query: '{user_query}'\nTranslated query: '{query_translator}'\n\nSummaries:\n{summaries_text}\n\nPlease provide the IDs of the most relevant summaries as JSON object separated by commas."
                }
            ],
            model="llama-3.1-70b-versatile",
            response_format={"type": "json_object"},
        )
        
        relevant_summary_ids = EnrichedQuery.model_validate_json(chat_completion.choices[0].message.content)
        return relevant_summary_ids
    
    async def summarize_final_response(self,relevant_response_list, query_translator, user_query):
        summaries_with_ids = [f"Summary {i}: {summary}" for i, summary in enumerate(relevant_response_list)]
        summaries_text = "\n\n".join(summaries_with_ids)
        
        chat_completion = await self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are given a list of enumerated summaries. Based on the user question, your task is to summarize all the provided summaries. Make sure that your answer is relevant to the user query."
                },
                {
                    "role": "user",
                    "content": f"User query: '{user_query}'\nTranslated query: '{query_translator}'\n\nSummaries:\n{summaries_text}\n\n. Answer: "
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()

    async def __call__(self,user_query:str):
        #Translate the simple user query to better query
        query_translator = await self.query_translator(user_query)
        
        relevant_summaries = self.summary_index.invoke(query_translator)
        
        summaries = ""
        for rs in relevant_summaries:
            summaries+=rs.page_content + "\n\n"
        
        #Based on relevant summaries, it translates the user query into three enriched queries
        enriched_queries = await self.query_enrichment(user_query, query_translator, summaries)
        enriched_queries = enriched_queries.split("\n\n")
        relevant_img_results: list[QueryImgTuple] = []
        
        for eq in enriched_queries:
            relevant_imgs = self.vision_index.search(query=eq,k=1)
            relevant_img_results.append(
                QueryImgTuple(query=eq,image_base64=[(i['base64'], await self.groq_response(i['base64'],eq)) for i in relevant_imgs])
            )

        manager_response_list:list[str] = []
        for ri in relevant_img_results:
            for r in ri.image_base64:
                #append the second index
                manager_response_list.append(
                    r[1]
                )
        #Manager response
        relevant_response = await self.manager_response(manager_response_list,query_translator, user_query)
        relevant_response_list = []
        relevant_ids = [r.strip() for r in relevant_response.enriched_queries.split(",")]
        for rp in relevant_ids:
            relevant_response_list.append(
                manager_response_list[int(rp)]
            )
        final_response = await self.summarize_final_response(relevant_response_list, query_translator, user_query)
        return query_translator, summaries, relevant_img_results, relevant_response_list, manager_response_list, final_response
    
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
