"""
llm_client.py
-------------
Unified LLM wrapper for the Student AI Companion.

All text generation (Q&A, flashcards, quiz, summary, topic resolution,
teaching notes) routes through Ollama running locally. This eliminates
Gemini free tier token quota issues for generate_content calls.

Gemini is kept only for text-embedding-004 which has a separate and
much higher free tier quota and has no local alternative of equal quality.

Default model: llama3.2 (2GB, fast on Apple Silicon)
Switch to:     mistral  (4GB, higher quality, slower)
               llama3.1 (4GB, strong reasoning)

Change the model by setting OLLAMA_MODEL in your .env file.
"""

import os
import ollama
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def generate(prompt: str, model: str = None) -> str:
    """
    Sends a prompt to the local Ollama model and returns the response text.

    Args:
        prompt: The fully assembled prompt string.
        model:  Ollama model name. Defaults to OLLAMA_MODEL env var
                or 'llama3.2' if not set.

    Returns:
        Response text string, or empty string on failure.
    """
    model = model or _DEFAULT_MODEL
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Warning: LLM call failed. Model: {model}. Reason: {e}")
        print("Ensure Ollama is running and the model is installed.")
        print(f"Run: ollama pull {model}")
        return ""


def generate_json(prompt: str, model: str = None) -> str:
    """
    Sends a prompt expecting a JSON response from the local Ollama model.
    Adds an explicit instruction to return only valid JSON.

    Returns:
        Raw response string (caller is responsible for json.loads).
    """
    model = model or _DEFAULT_MODEL
    json_prompt = (
        prompt
        + "\n\nIMPORTANT: Return only valid JSON. "
        "No markdown, no code fences, no explanation."
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": json_prompt}],
            format="json"
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Warning: JSON LLM call failed. Model: {model}. Reason: {e}")
        return "{}"
