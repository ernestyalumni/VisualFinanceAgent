from groq import AsyncGroq
import base64
import os
from typing import List, Optional, Annotated
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# prompt = """Analyze the image and provide:
# 1. A summary of the information in the image in exactly 3 sentences.
# 2. A list of keywords about the information provided in the image.

# Format your response as a JSON object with 'summary' and 'keywords' keys. 
# The 'summary' should be a string, and 'keywords' should be an array of strings. 
# Keywords should be descriptive and have 2-3 words, almost like phrases.  
# for instance, 'decrease' is not a good keyword, but 'decrease in investment in the US' is a good keyword.
# So there should be more context in each keyword group
# Ensure the output is valid JSON without any markdown formatting."""

prompt = "Summarize briefly in 2-3 lines"

client = AsyncGroq(api_key=os.environ['GROQ_API_KEY'])

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Data model for LLM to generate
class SummaryResponse(BaseModel):
    summary: Annotated[str, "Summary of the image in 2-3 lines that briefly describes the image"]
    keywords: Annotated[List[str], "List of keywords mentioned in the page"]


async def get_summary(img):
    completion = await client.chat.completions.create(
    model="llama-3.2-11b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img)}",
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




# if __name__ == '__main__':
    # recipe = await get_summary("output_png2_files/jll-asia-pacific-capital-tracker-3q23 (1)/page_4.png")
    # summary = asyncio.run(get_summary(data['base_64']))
# print_recipe(recipe)