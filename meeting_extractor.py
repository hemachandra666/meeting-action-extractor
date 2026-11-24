import json
import os

from groq import APIConnectionError, APIStatusError, Groq


MODEL = "openai/gpt-oss-20b"
INSTRUCTIONS = (
    "Extract explicit action items. Ignore forecasts. Return only a JSON array. "
    "Each item must have task, owner, and due. Use null for missing details. "
    "If no specific person or team is assigned, set owner to null. Treat 'someone' as unassigned. "
    "Return one item per current commitment. When a later sentence explicitly "
    "changes the deadline for the same task, update that item with the new deadline. "
    "Write the task as a short, self-contained action, such as 'Send the launch checklist'. "
    "Put the deadline only in the due field. "
    "When a task names a recipient or destination, include it in the task description. "
    "Keep distinct actions separate, even when they concern the same document."
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
        content = reply.choices[0].message.content.strip()
        lines = content.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```json", "```"}
            and lines[-1].strip() == "```"
        ):
            content = "\n".join(lines[1:-1])
        actions = json.loads(content)
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