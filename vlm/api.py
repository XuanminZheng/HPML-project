from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

# 图片转 base64
with open("img/1.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="openbmb/MiniCPM-V-4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image content."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
    }]
)
print(response.choices[0].message.content)