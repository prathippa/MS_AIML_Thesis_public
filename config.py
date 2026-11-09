"""Provider configuration. Supply verified accessible model IDs before init.
No model availability, pricing or deterministic behaviour is assumed.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def model(label, provider, env):
    provider = os.getenv(f"HTS_{label}_PROVIDER", provider)
    return {
        "label": label,
        "provider": provider,
        "model": os.getenv(env, "").strip(),
        "api_key_env": API_KEY_ENV.get(provider, "GROQ_API_KEY"),
    }


MODEL_A = model("A", "anthropic", "HTS_MODEL_A")
MODEL_B = model("B", "openai", "HTS_MODEL_B")
JUDGE_MODEL = model("J", "openai", "HTS_JUDGE_MODEL")
# Intentionally minimal Anthropic options; verify live provider compatibility.
DECODING_BY_PROVIDER = {
    "anthropic": {"max_tokens": 8000},
    "groq": {"max_tokens": 8000, "temperature": 0.0, "seed": 7},
    # Newer OpenAI models reject max_tokens/temperature; keep only the token cap.
    "openai": {"max_completion_tokens": 8000},
}
PRIMARY_COVERAGE = 0.70
COVERAGE_POINTS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
HUMAN_VALIDATION_SIZE = 30
MAX_API_CALLS = 2000
