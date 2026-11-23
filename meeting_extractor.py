import json
import os

from groq import APIConnectionError, APIStatusError, Groq


MODEL = "openai/gpt-oss-20b"
INSTRUCTIONS = (
    "Extract explicit action items. Ignore forecasts. Return only a JSON array. "
    "Each item must have task, owner, and due. Copy the task sentence exactly. "
    "Use null for missing details."
)


def extract_actions(note):
    if not note.strip():
        return []

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in the server environment.")

    try:
        with Groq(api_key=api_key, timeout=60, max_retries=0) as client:
            reply = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": note},
                ],
            )
    except APIStatusError as error:
        raise ValueError(f"Groq returned HTTP {error.status_code}.") from error
    except APIConnectionError as error:
        raise ValueError("Could not reach Groq. Please try again.") from error

    if not reply.choices or not reply.choices[0].message.content:
        raise ValueError("Groq returned an empty response.")

    try:
        actions = json.loads(reply.choices[0].message.content)
    except json.JSONDecodeError as error:
        raise ValueError("Groq returned a response that could not be read as JSON.") from error

    if not isinstance(actions, list):
        raise ValueError("Expected a JSON array of actions.")

    for action in actions:
        if not isinstance(action, dict) or set(action) != {"task", "owner", "due"}:
            raise ValueError("An action had missing or unexpected fields.")

        if not isinstance(action["task"], str) or not action["task"].strip():
            raise ValueError("An action was missing its task description.")

        for field in ("owner", "due"):
            if action[field] is not None and not isinstance(action[field], str):
                raise ValueError(f"Unexpected value for {field}.")

    return actions