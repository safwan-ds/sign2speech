import logging
import os
from collections.abc import Sequence

logger = logging.getLogger(__name__)

from config.architecture import architecture
from config.config import QWEN_MODEL_PATH


def create_llm_backend() -> tuple[object | None, dict[str, str]]:
    """Return (callable_or_None, backend_meta).

    The callable accepts ``prompt: str`` plus llama_cpp-style keyword args
    and returns ``{choices: [{text: ...}]}``.  ``generate_reply`` consumes
    this callable without change.

    backend_meta is a flat dict with at least a ``type`` key, suitable for
    display in the GUI backend badge.
    """
    global _remote_backend_active
    if not architecture.llm.use_qwen_llm:
        _remote_backend_active = False
        return None, {"type": "disabled"}

    backend = architecture.llm.llm_backend.strip().lower()
    if backend not in {"local", "remote"}:
        logger.warning("Unknown architecture.llm.llm_backend=%r; falling back to local", architecture.llm.llm_backend)
        backend = "local"

    if backend == "remote":
        _remote_backend_active = True
        return _make_remote_llm()
    _remote_backend_active = False
    return _make_local_llm()


def _make_local_llm() -> tuple[object | None, dict[str, str]]:
    llm = load_qwen_model()
    if llm is None:
        return None, {"type": "local_qwen", "status": "unavailable"}
    return llm, {"type": "local_qwen", "model_path": QWEN_MODEL_PATH}


def _make_remote_llm() -> tuple[object | None, dict[str, str]]:
    url = architecture.llm.llm_remote_url
    api_key = architecture.llm.llm_remote_api_key
    model = architecture.llm.llm_remote_model

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI SDK is not installed. Install it with: pip install openai")
        return None, {"type": "remote", "status": "unavailable"}

    client = OpenAI(
        api_key=api_key,
        base_url=url,
    )

    def _remote_call(prompt: str, **kwargs) -> dict:
        system = kwargs.get("system_prompt", "")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                temperature=kwargs.get("temperature", architecture.llm.qwen_inference_temperature),
                max_tokens=kwargs.get("max_tokens", architecture.llm.llm_remote_max_tokens)
            )
            content = response.choices[0].message.content
            return {"choices": [{"text": content}]}
        except Exception as exc:
            logger.error("Remote LLM call failed (%s): %s", url, exc)
            return {}

    return _remote_call, {"type": "remote", "url": url, "model": model}


def load_qwen_model():
    if not architecture.llm.use_qwen_llm:
        return None
    if not os.path.exists(QWEN_MODEL_PATH):
        logger.warning(f"Qwen model not found at {QWEN_MODEL_PATH}")
        return None
    if architecture.llm.qwen_force_gpu and architecture.llm.qwen_n_gpu_layers == 0:
        raise ValueError(
            "architecture.llm.qwen_force_gpu is enabled but architecture.llm.qwen_n_gpu_layers is 0. "
            "Set architecture.llm.qwen_n_gpu_layers to -1 (all) or a positive value."
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
        n_ctx=architecture.llm.qwen_n_ctx,
        n_gpu_layers=architecture.llm.qwen_n_gpu_layers,
        n_batch=architecture.llm.qwen_n_batch,
        flash_attn=True,
        verbose=False,
    )


FEW_SHOT_MAPPINGS = """STANDARD TOKEN-TO-SENTENCE MAPPINGS:
Input: ben üniversite gitmek
Output: Ben üniversiteye gidiyorum.

Input: sen proje bitirmek
Output: Sen projeyi bitirdin.

Input: merhaba ben
Output: Merhaba, ben.

Input: ben iyi
Output: Ben iyiyim.

Input: sen nasıl
Output: Sen nasılsın?

Input: o gelmek
Output: O geliyor.

Input: benim ad ne
Output: Benim adım ne?

Input: ben yardım etmek
Output: Ben yardım ediyorum.

Input: sen istemek ne
Output: Sen ne istiyorsun?

Input: ben öğrenci
Output: Ben öğrenciyim.

Input: ben yemek
Output: Ben yiyorum.

Input: sen yemek
Output: Sen yiyorsun.

Input: o yemek
Output: O yiyor.

Input: ben yemek içmek
Output: Ben yiyorum, içiyorum.

Input: ben yemek içmek istemek
Output: Ben yemek ve içmek istiyorum.

Input: hoca iyi
Output: Hoca iyi.

Input: öğrenci iyi
Output: Öğrenci iyi."""

QWEN_SYSTEM_PROMPT = """Sen bir Türkçe dilbilgisi düzenleyicisisin. Ham kelime dizilerini dilbilgisi kurallarına uygun, doğal Türkçe cümlelere dönüştürürsün.

KESİN KURALLAR:
1. HER ZAMAN TÜRKÇE YANIT VER. İngilizce veya başka bir dil kullanma.
2. YENİ KELİME EKLEME. Sadece verilen kelimeleri kullan. "ve" bağlacı eklenebilir.
3. FİİLLERİ ÇEKİMLE. Mastar halindeki fiilleri (yapmak, gitmek, gelmek, yemek, içmek) özneye göre çekimle. Varsayılan olarak Şimdiki Zaman (-yor) kullan.
   - Düzensiz fiillere dikkat et: yemek → yiyor (yeriyor DEĞİL), demek → diyor (deyor DEĞİL)
   - "istemek" fiili diğer fiillerden önce gelirse istek anlamı katar: "ben yemek istemek" → "Ben yemek istiyorum."
4. EKLERİ DOĞRU KULLAN. İsmin hallerini (-e, -de, -den) ve iyelik eklerini (-ım, -in, -i) doğru uygula.
5. SADECE CÜMLEYİ YAZ. Açıklama, yorum veya ek metin yazma."""

LOCAL_QWEN_LANGUAGE_CHECK = {"I ", "We ", "You ", "He ", "She ", "They ", "It ", "The ", "A ", "An "}

# Compact version for remote API calls (no QWEN tokens, fewer words).
REMOTE_SYSTEM_PROMPT = (
    "You are a Turkish syntactic processor. "
    "Restructure raw word tokens into a grammatically correct Turkish sentence. "
    "Conjugate verbs, apply suffixes, use Present Continuous by default. "
    "Output ONLY the sentence. No explanations.\n\n"
    "Examples:\n"
    "Input: ben üniversite gitmek  →  Ben üniversiteye gidiyorum.\n"
    "Input: sen proje bitirmek  →  Sen projeyi bitirdin.\n"
    "Input: merhaba ben  →  Merhaba, ben."
)

# Flag to track which backend is active for prompt selection.
_remote_backend_active = False


def generate_reply(
    llm,
    gesture_text: str,
    language: str = "tr",
    context: Sequence[str] | None = None,
) -> str | None:
    if llm is None:
        return None

    if _remote_backend_active:
        context_lines = ""
        if context:
            compact_context = [item.strip() for item in context if item.strip()]
            if compact_context:
                context_lines = (
                    "\n\nÖNCEKİ ÇIKTILAR:\n"
                    + "\n".join(f"- {item}" for item in compact_context[-2:])
                )
        system_prompt = REMOTE_SYSTEM_PROMPT + context_lines
        prompt = f"Input: {gesture_text}\nOutput:"
        result = llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=architecture.llm.llm_remote_max_tokens,
            temperature=architecture.llm.qwen_inference_temperature,
        )
        if isinstance(result, dict) and "choices" in result:
            text = result["choices"][0]["text"].strip()
            if text.startswith("Output:"):
                text = text.replace("Output:", "").strip()
            return text if text else None
        return None

    # Local QWEN: Turkish-optimized prompt with QWEN tokens.
    context_lines = ""
    if context:
        compact_context = [item.strip() for item in context if item.strip()]
        if compact_context:
            context_lines = (
                "\n\nÖNCEKİ ÇIKTILAR:\n"
                + "\n".join(f"- {item}" for item in compact_context[-2:])
            )

    system_prompt = f"{FEW_SHOT_MAPPINGS}\n\n{QWEN_SYSTEM_PROMPT}{context_lines}"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nInput: {gesture_text}\nOutput:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        result = llm(
            prompt,
            max_tokens=architecture.llm.qwen_max_tokens,
            temperature=0.1,
            top_p=0.5,
            stop=["<|im_end|>"],
        )
        if isinstance(result, dict) and "choices" in result:
            text = result["choices"][0]["text"].strip()
            if text.startswith("Output:"):
                text = text.replace("Output:", "").strip()
            # Reject English responses from QWEN
            if text:
                if any(text.startswith(eng) for eng in LOCAL_QWEN_LANGUAGE_CHECK):
                    logger.warning(
                        "QWEN output classified as English, discarding: %r", text[:60]
                    )
                    return None
            return text if text else None
        return None
    except Exception as e:
        logger.error(f"Qwen inference error: {e}")
        return None


def generate_turkish_reply(llm, gesture_text: str) -> str | None:
    return generate_reply(llm, gesture_text, language="tr")
