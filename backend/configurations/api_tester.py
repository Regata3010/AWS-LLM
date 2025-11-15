# test_openai.py
import openai
from settings import settings

openai.api_key = settings.OPENAI_API_KEY

try:
    response = openai.ChatCompletion.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello"}],
        max_completion_tokens=10
    )
    print("OpenAI working:", response.choices[0].message.content)
except Exception as e:
    print(f"OpenAI failed: {e}")