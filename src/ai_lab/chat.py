import ollama

MODEL = "qwen3:8b"
SYSTEM_PROMPT = "You are a concise assistant."


def main() -> None:
    # The memory: a list of dicts, like List<Dictionary<string, string>> in C#
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_tokens = 0

    print(f"Chatting with {MODEL}. Type 'exit' to quit, '/reset' to clear memory.")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/reset":
            messages = messages[:1]  # keep only the system prompt
            print("(memory cleared)")
            continue

        messages.append({"role": "user", "content": user_input})

        # Send the FULL history every time; that's what memory is
        response = ollama.chat(model=MODEL, messages=messages, think=False)

        reply = response.message.content
        messages.append({"role": "assistant", "content": reply})

        prompt_tokens = response.prompt_eval_count or 0
        reply_tokens = response.eval_count or 0
        total_tokens += prompt_tokens + reply_tokens

        print(f"\nAssistant: {reply}")
        print(
            f"[tokens] prompt: {prompt_tokens} | reply: {reply_tokens} "
            f"| session total: {total_tokens} | messages: {len(messages)}"
        )


if __name__ == "__main__":  # runs main() only when executed directly, not when imported
    main()