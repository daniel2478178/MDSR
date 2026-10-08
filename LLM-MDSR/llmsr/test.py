import requests
import json
import os

URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = os.environ['API_KEY']
MODEL = "deepseek-v4-flash"


headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: hello"
        }
    ],
    "temperature": 0.0,
    "max_tokens": 100,
}


try:
    response = requests.post(
        URL,
        headers=headers,
        json=payload,
        timeout=60,
        verify=False,
    )

    print("=" * 80)
    print("STATUS CODE:")
    print(response.status_code)

    print("=" * 80)
    print("RAW RESPONSE TEXT:")
    print(response.text)

    print("=" * 80)

    try:
        data = response.json()

        print("PARSED JSON:")
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as e:
        print("JSON parse failed:")
        print(repr(e))

except Exception as e:
    print("REQUEST FAILED:")
    print(type(e).__name__)
    print(repr(e))