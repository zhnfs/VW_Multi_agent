from __future__ import annotations

import hashlib

from acid_agent.business_rules import (
    infer_subtype_by_rules,
    is_acid_job_sentence,
    score_sentence_confidence,
    split_candidate_sentences,
)
from acid_agent.models import AcidJobEvent, ReportRecord


class ExtractionAgent:
    def __init__(self, min_confidence: float = 0.45) -> None:
        self.min_confidence = min_confidence

    def extract(self, reports: list[ReportRecord]) -> list[AcidJobEvent]:
        results: list[AcidJobEvent] = []
        seen: set[str] = set()

        for report in reports:
            for sentence in split_candidate_sentences(report.report_text):
                if not is_acid_job_sentence(sentence):
                    continue

                subtype = infer_subtype_by_rules(sentence)
                confidence = score_sentence_confidence(sentence, subtype)
                if confidence < self.min_confidence:
                    continue

                dedupe_key = self._dedupe_key(report.report_id, sentence)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                results.append(
                    AcidJobEvent(
                        event_id=dedupe_key,
                        well_id=report.well_id,
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
