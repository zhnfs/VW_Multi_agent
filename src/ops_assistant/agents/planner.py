from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ops_assistant.models import AgentPlan, QueryIntent
from ops_assistant.prompts import PLANNER_PROMPT

ASSET_ID_PATTERN = re.compile(r"\b[A-Za-z]{1,6}[-_ ]?\d{1,8}[A-Za-z0-9]*\b")


class PlannerAgent:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def plan(self, question: str, explicit_asset_id: str | None = None) -> AgentPlan:
        if self.llm is not None:
            llm_plan = self._plan_with_llm(question=question, explicit_asset_id=explicit_asset_id)
            if llm_plan is not None:
                return llm_plan

        asset_id = explicit_asset_id or self._infer_asset_id(question)
        if not asset_id:
            raise ValueError("No asset_id detected. Provide an asset ID in the prompt or UI input.")

        return AgentPlan(intent=self._infer_intent(question), asset_id=asset_id)

    def _plan_with_llm(self, question: str, explicit_asset_id: str | None) -> AgentPlan | None:
        chain = PLANNER_PROMPT | self.llm
        try:
            response = chain.invoke({"question": question, "asset_id": explicit_asset_id or ""})
        except Exception:
            return None

        try:
            payload = json.loads(_message_content(response))
        except json.JSONDecodeError:
            return None

        intent_value = str(payload.get("intent", "")).lower().strip()
        asset_id = (
            str(payload.get("asset_id", "")).strip()
            or explicit_asset_id
            or self._infer_asset_id(question)
        )
        if not asset_id:
            return None

        if intent_value not in {item.value for item in QueryIntent}:
            intent_value = self._infer_intent(question).value

        return AgentPlan(intent=QueryIntent(intent_value), asset_id=asset_id)

    @staticmethod
    def _infer_intent(question: str) -> QueryIntent:
        q = question.lower()
        asks_count = any(token in q for token in ("how many", "count", "number of", "total"))
        asks_subtype = any(token in q for token in ("subtype", "sub type", "type", "category"))

        if asks_count and asks_subtype:
            return QueryIntent.BOTH
        if asks_subtype:
            return QueryIntent.SUBTYPES
        if asks_count:
            return QueryIntent.COUNT
        return QueryIntent.BOTH

    @staticmethod
    def _infer_asset_id(question: str) -> str | None:
        match = ASSET_ID_PATTERN.search(question)
        if not match:
            return None
        return match.group(0).replace(" ", "")


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(str(item) for item in content).strip()
    return str(content).strip()
