from __future__ import annotations

import streamlit as st

from acid_agent.config import AppConfig
from acid_agent.service import build_orchestrator

PRESET_QUESTIONS = [
    "How many acid jobs were performed for well {well_id}?",
    "What are the acid job subtypes for well {well_id}?",
    "For well {well_id}, give me both acid job count and subtype distribution.",
]


@st.cache_resource
def _orchestrator():
    return build_orchestrator()


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def main() -> None:
    st.set_page_config(page_title="Acid Job Intelligence", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Acid Job Multi-Agent Assistant")
    st.caption(
        "Databricks Streamlit app using LangChain multi-agent orchestration, "
        "Unity Catalog daily reports, and MLflow tracing."
    )

    config = AppConfig.from_env()
    with st.sidebar:
        st.header("Query Settings")
        well_id = st.text_input("Well ID", value="", placeholder="e.g. WELL-1001")
        st.text_input(
            "Unity Catalog Table",
            value=config.fully_qualified_reports_table,
            disabled=True,
        )
        st.markdown("Accuracy target (business): **95% count**, **95% subtype**")

    st.subheader("Preset Questions")
    preset_cols = st.columns(len(PRESET_QUESTIONS))
    for idx, question in enumerate(PRESET_QUESTIONS):
        text = question.format(well_id=well_id or "<well_id>")
        if preset_cols[idx].button(text, key=f"preset-{idx}", use_container_width=True):
            st.session_state["pending_question"] = text

    _render_chat_history()

    typed_prompt = st.chat_input("Ask a question about acid jobs for a well")
    prompt = typed_prompt or st.session_state.pop("pending_question", None)
    if not prompt:
        return

    _append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    if not well_id:
        response_text = "Provide a `Well ID` in the sidebar before submitting a question."
        _append_message("assistant", response_text)
        with st.chat_message("assistant"):
            st.markdown(response_text)
        return

    with st.chat_message("assistant"), st.spinner("Running multi-agent workflow..."):
        try:
            response = _orchestrator().answer(question=prompt, explicit_well_id=well_id)
        except Exception as exc:
            st.error(str(exc))
            _append_message("assistant", f"Error: {exc}")
            return

        st.markdown(response.render_answer())
        st.metric("Faithfulness", f"{response.faithful_score:.2%}")

        if response.events:
            st.markdown("Evidence snippets")
            st.dataframe(
                [
                    {
                        "report_id": event.report_id,
                        "report_date": event.report_date,
                        "subtype": event.subtype.value if event.subtype else "",
                        "confidence": round(event.confidence, 3),
                        "evidence": event.evidence_text,
                    }
                    for event in response.events
                ],
                use_container_width=True,
            )

        _append_message("assistant", response.render_answer())


if __name__ == "__main__":
    main()
