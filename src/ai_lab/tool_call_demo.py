import ollama
from ai_lab.tools import TOOLS, TOOL_FUNCTIONS

MODEL = "qwen3:8b"

def main() -> None:
    question = "What is the average salary in the Engineering department?"
    messages = [{"role": "user", "content": question}]

    # Step 1: model sees the tools and decides
    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=False)
    messages.append(response.message)  # keep its tool request in history

    if not response.message.tool_calls:
        print("No tool call. Model answered directly:", response.message.content)
        return

    # Step 2: we run what it asked for
    for call in response.message.tool_calls:
        name = call.function.name
        args = call.function.arguments  # already a dict
        print(f"[tool call] {name}({args})")
        result = TOOL_FUNCTIONS[name](**args)  # **args = spread dict into keyword args
        print(f"[tool result] {result}")
        messages.append({"role": "tool", "content": result, "tool_name": name})

    # Step 3: model answers using the result
    final = ollama.chat(model=MODEL, messages=messages, think=False)
    print(f"\nAnswer: {final.message.content}")

if __name__ == "__main__":
    main()