import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

print("API key prefix:", os.getenv("HYDE_API_KEY", "")[:3])
print("Base URL:", os.getenv("HYDE_BASE_URL"))
print("Model:", os.getenv("HYDE_MODEL"))

# Read configuration
api_key = os.getenv("HYDE_API_KEY")
base_url = os.getenv("HYDE_BASE_URL")
model = os.getenv("HYDE_MODEL")

# Validate configuration
if not api_key or not base_url or not model:
    raise ValueError("Missing GLM API configuration in .env")

# Initialize GLM client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Test API connection
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: GLM API WORKING"
            }
        ],
        max_tokens=50
    )

    print("\nSUCCESS!")
    print("Model:", response.model)
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("\nAPI ERROR:", str(e))
