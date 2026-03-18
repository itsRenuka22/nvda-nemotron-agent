from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

MODEL = os.getenv("NVIDIA_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")


def complete(messages: list[dict], reasoning: bool = True, stream: bool = False, **kwargs):
    """Call the Nemotron model.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        reasoning: Prepend system message to enable detailed thinking.
        stream: If True, return the raw stream iterator; otherwise return full text.
        **kwargs: Extra params forwarded to the API (temperature, max_tokens, etc.).
    """
    if reasoning:
        messages = [{"role": "system", "content": "detailed thinking on"}] + messages

    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=stream,
        **kwargs,
    )

    if stream:
        return response

    return response.choices[0].message.content


def ask(prompt: str, reasoning: bool = True, **kwargs) -> str:
    """Convenience wrapper for a single user prompt. Returns full text."""
    return complete([{"role": "user", "content": prompt}], reasoning=reasoning, stream=False, **kwargs)


def ask_stream(prompt: str, reasoning: bool = True, **kwargs):
    """Stream a single user prompt, printing tokens as they arrive."""
    stream = complete([{"role": "user", "content": prompt}], reasoning=reasoning, stream=True, **kwargs)
    result = []
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            result.append(token)
    print()
    return "".join(result)


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print("Testing connection...")
    reply = ask("Reply with exactly: connection ok", reasoning=False)
    print(reply)
