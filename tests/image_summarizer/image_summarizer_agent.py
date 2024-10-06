import json
import io
import base64
import requests
from PIL import Image
import os

def call_openai_api(image, model):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set"

    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        prompt = """Analyze the image and provide:
1. A summary of the information in the image in exactly 3 sentences.
2. A list of keywords about the information provided in the image.

Format your response as a JSON object with 'summary' and 'keywords' keys. 
The 'summary' should be a string, and 'keywords' should be an array of strings. 
Keywords should be descriptive and have 2-3 words, almost like phrases.  
for instance, 'decrease' is not a good keyword, but 'decrease in investment in the US' is a good keyword.
So there should be more context in each keyword group
Ensure the output is valid JSON without any markdown formatting."""

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()

        if 'choices' in response_json and len(response_json['choices']) > 0:
            content = response_json['choices'][0]['message']['content'].strip()
            # Parse the JSON content
            result = json.loads(content)
            return result
        else:
            return {"error": f"Unexpected response format - {response_json}"}

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def get_image(image_path):
    # Load the local image file
    img = Image.open(image_path).convert("RGB")
    return img

def process_image(image_path):
    image = get_image(image_path)
    result = call_openai_api(image, "gpt-4o-mini")
    return result

def main():
    # Assuming the image is in the same directory as the script
    image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    result = process_image(image_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()