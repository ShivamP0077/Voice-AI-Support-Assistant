from groq import Groq

from backend.config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant. "
    "Use ONLY the context provided below to answer the customer's question. "
    "If the context does not contain enough information to answer, say so politely "
    "and suggest the customer contact human support. "
    "Keep your responses concise, friendly, and professional."
)


def generate_response(user_query: str, context_chunks: list[str]) -> str:
    """Generate an LLM response using Groq with RAG context injection.

    Args:
        user_query: The customer's question (transcribed from audio).
        context_chunks: Relevant text chunks retrieved from Qdrant.

    Returns:
        The assistant's response text.
    """
    # Build context block from retrieved chunks
    context_block = "\n\n".join(
        f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    # Compose messages with system prompt + context + user query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"### Relevant Context:\n{context_block}\n\n"
                f"### Customer Question:\n{user_query}"
            ),
        },
    ]

    chat_completion = _client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.4,
        max_tokens=512,
    )

    return chat_completion.choices[0].message.content
