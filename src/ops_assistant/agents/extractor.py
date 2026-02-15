from __future__ import annotations

import hashlib

from ops_assistant.business_rules import (
    infer_category_by_rules,
    is_target_event_sentence,
    score_sentence_confidence,
    split_candidate_sentences,
)
from ops_assistant.models import EventRecord, ReportRecord


class ExtractionAgent:
    def __init__(self, min_confidence: float = 0.45) -> None:
        self.min_confidence = min_confidence

    def extract(self, reports: list[ReportRecord]) -> list[EventRecord]:
        results: list[EventRecord] = []
        seen: set[str] = set()

        for report in reports:
            for sentence in split_candidate_sentences(report.report_text):
                if not is_target_event_sentence(sentence):
                    continue

                subtype = infer_category_by_rules(sentence)
                confidence = score_sentence_confidence(sentence, subtype)
                if confidence < self.min_confidence:
                    continue

                dedupe_key = self._dedupe_key(report.report_id, sentence)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                results.append(
                    EventRecord(
                        event_id=dedupe_key,
                        asset_id=report.asset_id,
                        report_id=report.report_id,
                        report_date=report.report_date,
                        evidence_text=sentence,
                        subtype=subtype,
                        confidence=confidence,
                    )
                )

        return sorted(results, key=lambda item: (item.report_date, item.report_id, item.event_id))

    @staticmethod
    def _dedupe_key(report_id: str, sentence: str) -> str:
        token = f"{report_id}|{sentence.strip().lower()}".encode()
        return hashlib.sha1(token).hexdigest()[:16]
