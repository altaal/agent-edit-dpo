"""Use one identical text-to-tool protocol for training and both evaluations."""

import json

from aci_patch_agent.agent import SYSTEM_PROMPT
from aci_patch_agent.tools import TOOLS


PROTOCOL = '\nReturn exactly one JSON object: {"tool": "edit", "arguments": {"start": 1, "end": 2, "replacement": "..."}}. No prose or markdown. For test or submit use empty arguments. Available tools: ' + json.dumps(TOOLS)


def prompt_for(task):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task.id}\n{task.issue}\n\nsolution.py:\n{task.source}"}]


def text_messages(messages):
    result = []
    for message in messages:
        role, content = message["role"], message.get("content") or ""
        if role == "system":
            content += PROTOCOL
        elif role == "tool":
            role, content = "user", "Tool result:\n" + content
        elif message.get("tool_calls"):
            call = message["tool_calls"][0]["function"]
            content = json.dumps({"tool": call["name"], "arguments": json.loads(call["arguments"])})
        result.append({"role": role, "content": content})
    return result


def edit_response(task, source):
    return json.dumps({"tool": "edit", "arguments": {"start": 1, "end": len(task.source.splitlines()), "replacement": source}})


def parse_response(raw, call_id):
    # Accept only one complete JSON object; malformed text consumes an agent action.
    try:
        value = json.loads(raw.strip())
        if (not isinstance(value, dict) or set(value) != {"tool", "arguments"}
                or value["tool"] not in {"view", "edit", "test", "submit"}
                or not isinstance(value["arguments"], dict)):
            raise ValueError("invalid tool object")
        return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": value["tool"], "arguments": json.dumps(value["arguments"])}}]}
    except (ValueError, TypeError):
        return {"role": "assistant", "content": raw}
