import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from config import (
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


def generate_turkish_reply(llm, gesture_text: str) -> str | None:
    if llm is None:
        return None

    system_prompt = (
        "Sen bir işaret dili çevirmenisin. Amacın izole kelimeleri kurallı bir Türkçe cümleye dönüştürmektir.\n"
        "Kesin kurallar:\n"
        "1. Yalnızca Türkçe yanıt ver.\n"
        "2. Açıklama yapma, noktalama işareti (nokta, virgül vb.) kullanma.\n"
        "3. Verilen kelimelerin tamamını kullan, hiçbirini silme.\n"
        "4. Cümleyi kurallı yapmak için kelimelere gerekli dilbilgisi eklerini (zaman, şahıs, hal ekleri) ekleyebilirsin.\n"
        "5. Anlamı değiştirecek yeni kök kelimeler ekleme.\n\n"
        "Örnek 1:\n"
        "Kelimeler: ben okul gitmek\n"
        "Yanıt: Ben okula gittim.\n\n"
        "Örnek 2:\n"
        "Kelimeler: sen proje yapmak\n"
        "Yanıt: Sen proje yaptın."
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nKelimeler: {gesture_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        result = llm(
            prompt,
            max_tokens=QWEN_MAX_TOKENS,
            temperature=QWEN_INFERENCE_TEMPERATURE,
            stop=["<|im_end|>", "\n"],
        )
        if isinstance(result, dict) and "choices" in result:
            text = result["choices"][0]["text"].strip()
            return text if text else None
        return None
    except Exception as e:
        logger.error(f"Qwen inference error: {e}")
        return None
