import os
from groq import Groq
from PIL import Image
import requests
from io import BytesIO
import base64

def expand_query_more(user_query, model_name="llama-3.1-70b-versatile"):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    system_content = (
        "You are an expert in expanding investment and consulting queries. "
        "Your task is to take a brief user query and generate 3 expanded, "
        "well-formed questions that provide more context and depth. Focus on "
        "creating full sentences with proper grammar that explore different "
        "aspects of the original query. Do not write anything other than the "
        "expanded queries. Do not write 'here is the queries' or anything, "
        "just write the 3 queries, and nothing else."
    )

    user_content = \
        "Please expand the following query into 3 more detailed questions: '{user_query}'"

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

    expanded_queries = chat_completion.choices[0].message.content.split('\n')
    return expanded_queries

def get_image_more(query):
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

    system_content = (
        "You are an expert in evaluating image relevance to queries. Your task "
        "is to determine if the given image is relevant to the user's query. "
        "Respond with 'OK' if the image is relevant, and 'NO' if it's not "
        "relevant. Do not provide any other explanation or text."
    )

    user_content = (
        "Is this image relevant to the query: '{query}'? Image: {img_str}"
    )

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

def main():
    user_query = "asia real estate investments"

    image = get_image(user_query)
    relevance = evaluate_image_relevance(user_query, image)

    print(f"\nImage relevance: {relevance}")

if __name__ == "__main__":
    main()