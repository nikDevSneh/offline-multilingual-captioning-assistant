"""
Optional Hindi <-> English translation layer, applied after ASR when
translation.enabled is true in config.yaml.

Kept as a swappable interface: NoOpTranslator (default) does nothing, so the
core captioning pipeline works with zero extra dependencies. TransformersTranslator
loads a small, locally-runnable MT model the first time it's used.
"""
from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        ...


class NoOpTranslator:
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return text


class TransformersTranslator:
    """
    Lazily loads a Hugging Face translation model on first use. Swap
    `model_name` in config.yaml for any locally-runnable MT model, e.g.
    an IndicTrans2 or NLLB-distilled checkpoint appropriate for Hindi<->English.
    """

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self._pipeline = None  # loaded on first translate() call

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline  # lazy import

        self._pipeline = pipeline("translation", model=self.model_name)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return text
        self._ensure_loaded()
        # NLLB-style models expect language codes like "hin_Deva"/"eng_Latn";
        # map simple "hi"/"en" config values to those before calling out.
        lang_map = {"hi": "hin_Deva", "en": "eng_Latn"}
        src = lang_map.get(source_lang, source_lang)
        tgt = lang_map.get(target_lang, target_lang)
        result = self._pipeline(text, src_lang=src, tgt_lang=tgt)
        return result[0]["translation_text"]


def build_translator(cfg: dict) -> Translator:
    if not cfg.get("enabled", False):
        return NoOpTranslator()
    return TransformersTranslator(model_name=cfg.get("model_name", "facebook/nllb-200-distilled-600M"))
