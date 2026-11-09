"""Central configuration. Secrets come from the environment (.env), never hard-coded."""
import os
from dotenv import load_dotenv
load_dotenv()

# ---- Models (Decision 3) --------------------------------------------------
# Exact snapshot IDs and access dates are recorded at run time for reproducibility.
MODEL_A = {
    "label": "Model A",
    "provider": "anthropic",
    "model": "claude-sonnet-5-5",     # Claude Sonnet 5.5 API string; confirm/record exact snapshot at run time
    "api_key_env": "ANTHROPIC_API_KEY",
}
MODEL_B = {
    "label": "Model B",
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",   # Meta Llama 3.3 70B, hosted on Groq
    "api_key_env": "GROQ_API_KEY",
}

# ---- Decoding (fixed and reported; identical across models) ----------------
DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 600, "seed": 7}
RETRY_ONCE_ON_MALFORMED = True

# ---- Dataset splits (ATLAS/CROSS; predefined by the authors) ---------------
SPLITS = {"train": 18254, "validation": 200, "test": 200}
DATA_DIR = "data"
RESULTS_DIR = "results"

# ---- Routing experiment (Decision 1 + Stage 7) -----------------------------
PRIMARY_COVERAGE = 0.70
COVERAGE_POINTS = [0.50, 0.60, 0.70, 0.80, 0.90]

# ---- Evidence-support study (Stage 5 + Decision 5) -------------------------
JUDGE_SUBSET_SIZE = 60          # 4 strata x ~15; fallback 30 if constrained
HUMAN_VALIDATION_SIZE = 30      # mandatory independent second-annotator check (Cohen's kappa)

# ---- Cost guardrails -------------------------------------------------------
MAX_API_CALLS = 2000            # hard cap so a bug cannot run up charges
CACHE_RESPONSES = True          # cache + timestamp every model output
