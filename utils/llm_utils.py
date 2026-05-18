import logging
import os

logger = logging.getLogger(__name__)

from config.config import (
    USE_QWEN_LLM,
    QWEN_MODEL_PATH,
    QWEN_N_CTX,
    QWEN_N_GPU_LAYERS,
    QWEN_N_BATCH,
    QWEN_FORCE_GPU,
    QWEN_MAX_TOKENS,
    QWEN_INFERENCE_TEMPERATURE,
)


def load_qwen_model():
    if not USE_QWEN_LLM:
        return None
    if not os.path.exists(QWEN_MODEL_PATH):
        logger.warning(f"Qwen model not found at {QWEN_MODEL_PATH}")
        return None
    if QWEN_FORCE_GPU and QWEN_N_GPU_LAYERS == 0:
        raise ValueError(
            "QWEN_FORCE_GPU is enabled but QWEN_N_GPU_LAYERS is 0. "
            "Set QWEN_N_GPU_LAYERS to -1 (all) or a positive value."
        )
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error(
            "llama-cpp-python is not installed. "
            "Install it with: pip install llama-cpp-python"
        )
        return None
    return Llama(
        model_path=QWEN_MODEL_PATH,
        n_ctx=QWEN_N_CTX,
        n_gpu_layers=QWEN_N_GPU_LAYERS,
        n_batch=QWEN_N_BATCH,
        flash_attn=True,
        verbose=False,
    )


def generate_reply(llm, gesture_text: str, language: str = "tr") -> str | None:
    if llm is None:
        return None

    system_prompt = """You are a syntactic processing module for a sign language translation array. Your objective is to take a sequential array of raw word tokens (often in base or infinitive forms) and restructure them into a grammatically correct, natural-sounding sentence.

    STRICT CONSTRAINTS:
    1. NO NEW VOCABULARY: You are strictly forbidden from adding new semantic concepts or nouns. Use only the concepts provided.
    2. CONJUGATE VERBS: If a verb is provided in the infinitive form (e.g., "yapmak", "gitmek", "go"), you MUST conjugate it to match the subject. DEFAULT to the Present Continuous tense (e.g., "yapıyorum", "am doing") unless the context clearly implies past tense.
    3. APPLY SUFFIXES: Apply necessary grammatical modifications (declension, case markers, prepositions) to the existing nouns to ensure proper syntax (e.g., "üniversite" -> "üniversiteye").
    4. OUTPUT FORMAT: Output the finalized sentence string ONLY. Do not output explanations, conversational filler, or markdown delimiters.

    EXAMPLES:
    Input: me university go
    Output: I am going to the university.

    Input: ben üniversite gitmek
    Output: Ben üniversiteye gidiyorum.

    Input: sen proje bitirmek
    Output: Sen projeyi bitirdin.

    Input: merhaba ben eldiven yapmak
    Output: Merhaba, ben eldiven yapıyorum."""

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nInput: {gesture_text}\nOutput:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        result = llm(
            prompt,
            max_tokens=QWEN_MAX_TOKENS,
            temperature=QWEN_INFERENCE_TEMPERATURE,
            top_p=0.5,
            stop=["<|im_end|>", "\n"],
        )
        if isinstance(result, dict) and "choices" in result:
            text = result["choices"][0]["text"].strip()
            # 3. Clean up the output in case the model repeats the trigger word
            if text.startswith("Output:"):
                text = text.replace("Output:", "").strip()
            return text if text else None
        return None
    except Exception as e:
        logger.error(f"Qwen inference error: {e}")
        return None


def generate_turkish_reply(llm, gesture_text: str) -> str | None:
    return generate_reply(llm, gesture_text, language="tr")
