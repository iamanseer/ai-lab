from datetime import datetime
import json

PEOPLE = {
    "Alice": {"age": 30, "city": "New York"},
    "Bob": {"age": 25, "city": "Los Angeles"},
}


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def get_time() -> str:
    """Get the current time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def lookup(key: str) -> dict:
    """Look up a person's age and city by name."""
    return PEOPLE[key]


TOOLS = {
    "add": add,
    "get_time": get_time,
    "lookup": lookup,
}


def dispatch(name: str, args: dict) -> str:
    """Run a tool by name with keyword args; return result or error as text."""
    if name not in TOOLS:
        return f"error: unknown tool '{name}'"

    func = TOOLS[name]
    try:
        result = func(**args)
        return str(result)
    except TypeError as e:
        return f"error: bad arguments for '{name}': {e}"
    except KeyError as e:
        return f"error: not found: {e}"


MODEL_OUTPUTS = [  
    '{"tool": "add", "args": {"a": 2, "b": 3}}'
    '{"tool": "get_time", "args": {}}'
    '{"tool": "lookup", "args": {"key": "Bob"}}'
    '{"tool": "multiply", "args": {"a": 2, "b": 3}}'
    '{"tool": "add", "args": {"a": 2}}'
    '{"tool": "add", "args": {"a": 2, "b": 3}' 
]

def handle(raw: str) -> str:
    """Handle a raw model output string, returning the tool result or error."""
    try:
        data = json.loads(raw)
        tool_name = data["tool"]
        args = data.get("args", {})
        return dispatch(tool_name, args)
    except json.JSONDecodeError as e:
        return f"error: invalid JSON: {e}"
    except KeyError as e:
        return f"error: missing key in input: {e}"


for raw in MODEL_OUTPUTS:
    result = handle(raw)
    print(f"Input: {raw}\nOutput: {result}\n")

    