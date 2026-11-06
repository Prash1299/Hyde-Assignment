
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["HYDE_API_KEY"],
    base_url=os.environ["HYDE_BASE_URL"],
    max_retries=0
)

try:
    result = client.chat.completions.create(
        model="hyde/glm-5.3-flash",
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=20
    )

    print("SUCCESS:", result.choices[0].message.content)

except Exception as e:
    print("Exception:", type(e).__name__)
    print("HTTP status:", getattr(e, "status_code", None))
    print("Details:", str(e))
