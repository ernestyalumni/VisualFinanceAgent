import streamlit as st
from pipeline import VisionFinancePipeLine, EnrichedQuery, QueryImgTuple
import asyncio
from io import BytesIO
import numpy as np
from PIL import Image
import base64

@st.cache_resource
def get_pipeline():
    vfsp = VisionFinancePipeLine()
    
    return vfsp

vfsp = get_pipeline()
st.session_state['vfsp'] = vfsp

def convert_base64_to_img(base64_str:str):
    base64_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(base64_data))
    return image

async def query_translator_ans(user_query):
    query_translator = await vfsp.query_translator(user_query)
    return query_translator

async def query_enrich(query_translator, user_query):
    relevant_summaries = vfsp.summary_index.invoke(query_translator)

    summaries = ""
    for rs in relevant_summaries:
        summaries+=rs.page_content + "\n\n"

    #Based on relevant summaries, it translates the user query into three enriched queries
    enriched_queries = await vfsp.query_enrichment(user_query, query_translator, summaries)
    enriched_queries = enriched_queries.split("\n\n")
    return enriched_queries

async def response_from_images(enriched_queries):
    relevant_img_results: list[QueryImgTuple] = []

    for eq in enriched_queries:
        relevant_imgs = vfsp.vision_index.search(query=eq,k=2)
        relevant_img_results.append(
            QueryImgTuple(query=eq,image_base64=[(i['base64'], await vfsp.groq_response(i['base64'],eq)) for i in relevant_imgs])
        )

    manager_response_list:list[str] = []
    for ri in relevant_img_results:
        for r in ri.image_base64:
            #append the second index
            manager_response_list.append(
                r[1]
            )
    return manager_response_list, relevant_img_results
#Manager response
async def get_manager_response(manager_response_list,query_translator, user_query):
    relevant_response = await vfsp.manager_response(manager_response_list,query_translator, user_query)
    st.write(f"Relevant Response: {relevant_response}")
    # relevant_response = EnrichedQuery.model_validate_json(relevant_response)
    relevant_response_list = []
    relevant_ids = [r.strip() for r in relevant_response.split(",")]
    for rp in relevant_ids:
        relevant_response_list.append(
            manager_response_list[int(rp)]
        )
    return relevant_response_list

async def get_final_answer(relevant_response_list, query_translator, user_query):
    final_response = await vfsp.summarize_final_response(relevant_response_list, query_translator, user_query)
    return final_response

user_input = st.text_input("You:", key="user_input")

async def main(user_input):
    with st.spinner('Query Translation'):
        query_translator = await query_translator_ans(user_input)

    st.write(f"Translated Query: {query_translator}")
    
    with st.spinner('Query Translation'):

        enriched_queries = await query_enrich(query_translator, user_input)
    
    st.write(f"Enriched Query: {enriched_queries}")
    
    with st.spinner("Answer from Images"):
        manager_response_list, relevant_img_results = await response_from_images(enriched_queries)
    
    with st.spinner("Filter Responses"):
        relevant_response_list = await get_manager_response(manager_response_list,query_translator, user_input)
    
    with st.spinner("Final Answer"):
        final_answer = await get_final_answer(relevant_response_list,query_translator,user_input)
    
    st.write(f"Final Answer: {final_answer}")
    
    for rir in relevant_img_results:
        st.write("Query: ",rir.query)
        for rer in rir.image_base64:
            # print(rer)
            st.image(convert_base64_to_img(rer[0]))
            # st.write("Answer: ",rer[1])

if user_input:
    asyncio.run(main(user_input))
