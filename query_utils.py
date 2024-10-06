from groq import Groq
import os

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def query_translator(user_query: str) -> str:
    chat_completion = groq_client.chat.completions.create(
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
    return chat_completion.choices[0].message.content.strip()

def query_enrichment(user_query: str, translated_query: str, summaries: str) -> list:
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in generating enriched queries based on original queries, translated queries, and relevant summaries. Your task is to generate 3 enriched queries that explore different aspects of the topic. Return only the list of 3 enriched queries, separated by newlines."
            },
            {
                "role": "user",
                "content": f"Original query: '{user_query}'\nTranslated query: '{translated_query}'\nRelevant summaries: {summaries}\n\nPlease generate 3 enriched queries based on this information."
            }
        ],
        model="llama-3.1-70b-versatile",
    )
    return chat_completion.choices[0].message.content.strip().split('\n')


