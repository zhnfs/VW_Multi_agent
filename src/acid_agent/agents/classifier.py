from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from acid_agent.models import AcidJobEvent, AcidSubtype
from acid_agent.prompts import CLASSIFIER_PROMPT, load_sme_business_rules


class ClassificationAgent:
    def __init__(self, llm: BaseChatModel | None = None, llm_threshold: float = 0.70) -> None:
        self.llm = llm
        self.llm_threshold = llm_threshold
        self.rules = load_sme_business_rules()

    def classify(self, events: list[AcidJobEvent]) -> list[AcidJobEvent]:
        classified: list[AcidJobEvent] = []
        for event in events:
            updated_event = event

            if self.llm is not None and event.confidence < self.llm_threshold:
                llm_subtype = self._classify_with_llm(event.evidence_text)
                if llm_subtype is not None:
                    updated_event = event.model_copy(
                        update={
                            "subtype": llm_subtype,
                            "confidence": min(1.0, event.confidence + 0.15),
                        }
                    )

            # Safety default to one of the 4 allowed subtype labels.
            if updated_event.subtype is None:
                updated_event = updated_event.model_copy(
                    update={"subtype": AcidSubtype.MATRIX_ACIDIZING}
                )

            classified.append(updated_event)

        return classified

    def _classify_with_llm(self, evidence: str) -> AcidSubtype | None:
        chain = CLASSIFIER_PROMPT | self.llm
        try:
            response = chain.invoke({"rules": self.rules, "evidence": evidence})
        except Exception:
            return None

        try:
            payload = json.loads(_message_content(response))
        except json.JSONDecodeError:
            return None

        subtype_value = str(payload.get("subtype", "")).strip().lower()
        try:
            return AcidSubtype(subtype_value)
        except ValueError:
            return None


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item) for item in content).strip()
    return str(content).strip()
