import os
from groq import Groq
from PIL import Image
import requests
from io import BytesIO
import base64
from typing import List

def expand_query(user_query):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in expanding investment and consulting queries. Your task is to take a brief user query and generate 3 expanded, well-formed questions that provide more context and depth. Focus on creating full sentences with proper grammar that explore different aspects of the original query. Do not write anything other than the expanded queries. Do not write 'here is the queries' or anything, just write the 3 queries, and nothing else."
            },
            {
                "role": "user",
                "content": f"Please expand the following query into 3 more detailed questions: '{user_query}'"
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    expanded_queries = chat_completion.choices[0].message.content.split('\n')
    return expanded_queries

def get_image(query):
    # Load the local image file
    image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    img = Image.open(image_path)
    return img

def evaluate_image_relevance(query, image):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Convert image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in evaluating image relevance to queries. Your task is to determine if the given image is relevant to the user's query. Respond with 'OK' if the image is relevant, and 'NO' if it's not relevant. Do not provide any other explanation or text."
            },
            {
                "role": "user",
                "content": f"Is this image relevant to the query: '{query}'? Image: {img_str}"
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    relevance = chat_completion.choices[0].message.content.strip()
    return relevance

def evaluate_multiple_text_relevance_basic(
    user_query: str,
    texts: List[str],
    query_translator: str,
    model_name="llama-3.1-70b-versatile",
    ):
    summaries_with_ids = \
        [f"Summary {i}: {summary}" for i, summary in enumerate(texts)]
    summaries_text = "\n\n".join(summaries_with_ids)

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    #system_content = f"You are an expert in evaluating the relevance of information to user queries. Your task is to analyze a list of summaries and determine which ones are most relevant to the user's original query and the translated query. Return only the IDs of the most relevant summaries as JSON object, separated by commas."
    system_content = f"You are an expert in evaluating the relevance of information to user queries. Your task is to analyze a list of summaries and determine which ones are most relevant to the user's original query and the translated query. Return only the IDs of the most relevant summaries separated by commas."
    # system_content = (
    #     "You are an expert in evaluating the relevance of information to user "
    #     "queries. Your task is to analyze a list of summaries and determine "
    #     "which ones are most relevant to the user's original query and the "
    #     "translated query. Return only the IDs of the most relevant summaries, "
    #     "separated by commas."
    # )

    #user_content = f"User query: '{user_query}'\nTranslated query: '{query_translator}'\n\nSummaries:\n{summaries_text}\n\nPlease provide the IDs of the most relevant summaries as JSON object separated by commas."
    user_content = f"User query: '{user_query}'\nTranslated query: '{query_translator}'\n\nSummaries:\n{summaries_text}\n\nPlease provide the IDs of the most relevant summaries separated by commas."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        model=model_name,
    )

    relevant_summary_ids = chat_completion.choices[0].message.content.strip()
    relevant_summary_ids_cleaned =[int(id.strip()) for id in relevant_summary_ids.split(',')]
    output_summaries = [texts[i] for i in relevant_summary_ids_cleaned]
    return output_summaries


def evaluate_multiple_text_relevance(
    query: str,
    texts: List[str],
    model_name="llama-3.1-70b-versatile"):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Aggregate texts into a single string
    aggregated_text = " ".join(texts)

    system_content = (
        "You are an expert in evaluating multiple pieces of text for relevance. "
        "You are also an expert in composing text summaries that only includes "
        "important points, directly relevant details to a given query and "
        "nothing else at all. You do not provide any other information. Your "
        "task is to take a given text that is an aggregation of multiple texts, "
        "and compose a single text summary that is relevant to the user's query. "
        "Do not provide any other explanation or text for anything with little "
        "or no relevancy to the query."
    )

    user_content = \
        f"Summarize this text with statements only relevant to the query: '{query}'? Text: '{aggregated_text}'"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        model=model_name,
    )

    summary = chat_completion.choices[0].message.content.strip()
    return summary

def main():
    user_query = "asia real estate investments"

    image = get_image(user_query)
    relevance = evaluate_image_relevance(user_query, image)

    print(f"\nImage relevance: {relevance}")

if __name__ == "__main__":
    main()