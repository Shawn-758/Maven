"""
rag/generator.py
----------------
COMPONENT 6 & 7: Prompt Construction + Grounded Response Generation

What it does:
  1. Builds a structured prompt that injects retrieved context into the
     LLM's context window
  2. Calls the LLM API and returns its response

This is where "grounding" happens:
  Instead of asking the LLM to answer from its training data (which can
  hallucinate), we give it specific passages and instruct it to answer
  ONLY from those passages. This is the core value proposition of RAG.

How prompt construction affects quality:
  The prompt template is arguably the most impactful single parameter
  in a RAG system. A few key principles:

  1. State the task clearly: "You are an assistant. Answer ONLY from context."
  2. Format context clearly: number the passages so the LLM can reference them
  3. Instruct on failure: "If the answer is not in the context, say so."
  4. Include the question at the END: models attend more to recent tokens.

LLM choices:
  This module supports:
  - Anthropic Claude (default, what we use here)
  - OpenAI GPT-4 / GPT-3.5 (drop-in swap)
  - Open-source via Ollama (runs locally, free, no API key)

  The interface is the same regardless of backend — swap the client,
  not your whole codebase.
"""

from .vector_store import SearchResult


# ------------------------------------------------------------------ #
# Prompt templates                                                      #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are a precise question-answering assistant.

Your job:
- Answer questions ONLY using the provided context passages.
- If the answer is not found in the context, say "I don't have enough information in the provided context to answer this question." Do NOT use your general training knowledge.
- Cite which passage(s) you used by referencing [Passage N].
- Be concise and direct. Avoid unnecessary padding.
- If multiple passages are relevant, synthesize them into a coherent answer."""


def build_prompt(query: str, search_results: list[SearchResult]) -> str:
    """
    Construct the user message that combines context + question.

    Structure:
      CONTEXT PASSAGES
      [Passage 1]: <text> (source: <filename>)
      ...
      QUESTION
      <query>

    Why number the passages?
      Numbered references allow the LLM to say "According to [Passage 2]..."
      which makes it easy to verify the source. This is the foundation of
      source attribution / citation in RAG systems.
    """
    if not search_results:
        return f"QUESTION:\n{query}\n\nNote: No relevant context was retrieved."

    context_block = "\n\n".join(
        f"[Passage {r.rank}] (source: {r.metadata.get('source', 'unknown')}, "
        f"similarity: {r.score:.2f}):\n{r.text}"
        for r in search_results
    )

    return (
        f"CONTEXT PASSAGES:\n"
        f"{context_block}\n\n"
        f"{'─' * 60}\n\n"
        f"QUESTION:\n{query}"
    )


# ------------------------------------------------------------------ #
# LLM clients                                                           #
# ------------------------------------------------------------------ #

class AnthropicGenerator:
    """
    Uses Anthropic's Claude API to generate answers.

    Requires: pip install anthropic
    API key:  set ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, query: str, search_results: list[SearchResult]) -> str:
        user_message = build_prompt(query, search_results)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OpenAIGenerator:
    """
    Uses OpenAI's API to generate answers.

    Requires: pip install openai
    API key:  set OPENAI_API_KEY environment variable
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def generate(self, query: str, search_results: list[SearchResult]) -> str:
        user_message = build_prompt(query, search_results)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content


class OllamaGenerator:
    """
    Uses Ollama to run open-source LLMs locally.

    No API key needed. Requires Ollama installed and running:
      curl https://ollama.ai/install.sh | sh
      ollama pull llama3.2

    Great for: privacy-sensitive use cases, offline use, cost-free experimentation.
    Trade-off: slower than hosted APIs unless you have a GPU.
    """

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, query: str, search_results: list[SearchResult]) -> str:
        import requests

        user_message = build_prompt(query, search_results)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"

        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]