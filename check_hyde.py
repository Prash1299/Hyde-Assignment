
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("HYDE_API_KEY"),
    base_url="https://inference.hyde.tech/v1"
)

try:
    print("Checking Hyde authentication...")

    models = client.models.list()

    print("Authentication successful!")
    print("Available models:")

    for model in models.data:
        print("-", model.id)

except Exception as e:
    print("Authentication failed:")
    print(e)
