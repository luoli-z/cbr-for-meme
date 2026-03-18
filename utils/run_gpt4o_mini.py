# run_gpt4o_mini.py
from openai import OpenAI
import base64
import os
import time
#os.environ["OPENAI_API_KEY"] = "your_api_key"  
'''client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)'''

def encode_image(image_path):
       with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_gpt4o_mini_response(prompt: str, client, image_path: str = None):

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if image_path and os.path.exists(image_path):
        base64_img = encode_image(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            }
        )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
            model="qwen-plus",  
            messages=messages,
            temperature=0,  
            max_tokens=1024,  
            )
            result = response.choices[0].message.content.strip()
            break
        except:
            if attempt < 2:
                continue
            result = "Updated rules:Harmful memes involve discrimination or illegality;normal ones are harmless."
    return result
def get_gpt4o_mini_response2(prompt: str, client, image_path: str = None):

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if image_path and os.path.exists(image_path):
        base64_img = encode_image(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            }
        )

    response = client.chat.completions.create(
        model="gemini-2.0-flash", 
        messages=messages,
        temperature=0, 
        max_tokens=1024, 
    )
    return response.choices[0].message.content.strip()