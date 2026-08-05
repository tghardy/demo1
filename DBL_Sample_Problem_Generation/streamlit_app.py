"""Streamlit frontend for generation + traversal tutoring."""

from __future__ import annotations

import json

import streamlit as st  # pyright: ignore[reportMissingImports]

from dbl_problem_generator.practice_generator import PracticeProblemGenerator
from dbl_problem_generator.llm_factory import build_chat_model
from dbl_problem_generator.traversal_sampler import (
    JointTraversal,
    Neo4jTraversalSampler,
    SampleTraversal,
    TraversalSampler,
)
from dbl_student_rag import StudentTutor


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return "gpt-4o-mini"
    return "qwen2.5:14b"


def _init_state() -> None:
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "ollama"
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = _default_model_for_provider(
            str(st.session_state.llm_provider)
        )
    if "generation_result" not in st.session_state:
        st.session_state.generation_result = None
    if "tutor" not in st.session_state:
        st.session_state.tutor = None
    if "tutor_result" not in st.session_state:
        st.session_state.tutor_result = None
    if "loaded_traversal" not in st.session_state:
        st.session_state.loaded_traversal = None


def _sync_model_default() -> None:
    st.session_state.llm_model_name = _default_model_for_provider(
        str(st.session_state.llm_provider)
    )


def _load_tree_for_model(model_id: int) -> dict[str, dict]:
    sampler = Neo4jTraversalSampler.from_environment()
    try:
        query = """
        MATCH (m:Model {id: $model_id})<-[:BELONGS_TO]-(n:Decision)
        OPTIONAL MATCH (n)-[r:HAS_CHOICE]->(child:Decision)
        RETURN n.id AS node_id, n.text AS question, r.text AS answer_text, child.id AS child_id
        """
        with sampler.driver.session() as session:
            records = list(session.run(query, model_id=model_id))
    finally:
        sampler.close()

    nodes: dict[str, dict] = {}
    for record in records:
        node_id = str(record["node_id"])
        if node_id not in nodes:
            nodes[node_id] = {
                "node_id": node_id,
                "question": str(record["question"] or ""),
                "options": [],
            }
        if record["answer_text"] is not None and record["child_id"] is not None:
            nodes[node_id]["options"].append(
                {
                    "answer_text": str(record["answer_text"]),
                    "next_node_id": str(record["child_id"]),
                }
            )

    if not nodes:
        raise ValueError(f"No decision nodes found for model {model_id}.")
    return nodes


def _path_display(tutor: StudentTutor) -> str:
    state = tutor.state
    parts = [state["node_history"][0]]
    for answer, node_id in zip(state["answer_history"], state["node_history"][1:]):
        parts.append(f"--[{answer}]--> {node_id}")
    return " ".join(parts)


def _render_current_step(tutor: StudentTutor) -> str:
    state = tutor.state
    node = state["nodes_by_id"][state["current_node_id"]]
    message = f"Current question: {node['question']}"
    if node["options"]:
        message += "\n\nOptions:\n" + "\n".join(f"- {o['answer_text']}" for o in node["options"])
    return message


def _build_tutor_result(tutor: StudentTutor) -> dict:
    state = tutor.state
    node = state["nodes_by_id"][state["current_node_id"]]
    return {
        "response": state["response"],
        "completed": state["completed"],
        "current_question": node["question"],
        "current_options": [o["answer_text"] for o in node["options"]],
        "path_display": _path_display(tutor),
        "chat_history": state["chat_history"],
    }


def _build_traversal_from_payload(payload: dict) -> SampleTraversal | JointTraversal:
    traversals_payload = payload.get("traversals")
    if isinstance(traversals_payload, list):
        return JointTraversal(
            [TraversalSampler.from_payload(item) for item in traversals_payload]
        )

    if isinstance(payload.get("traversal"), dict):
        return TraversalSampler.from_payload(payload["traversal"])

    return TraversalSampler.from_payload(payload)


def _generate_problem(
    model_id: int,
    strategy: str,
    provider: str,
    model_name: str,
    temperature: float,
    traversal: SampleTraversal | JointTraversal | None = None,
) -> None:
    if traversal is None:
        sampler = Neo4jTraversalSampler.from_environment()
        try:
            sampler.verify_connectivity()
            traversal = (
                sampler.sample_balanced_leaf(model_id=model_id)
                if strategy == "balanced-leaf"
                else sampler.sample_from_all_paths(model_id=model_id)
            )
        finally:
            sampler.close()

    llm = build_chat_model(provider=provider, model_name=model_name, temperature=temperature)
    generator = PracticeProblemGenerator(llm=llm)
    result = generator.generate(traversal)

    nodes_by_id = _load_tree_for_model(model_id=model_id)
    traversal_paths = result.get("traversal_paths", [result["traversal_path"]])
    target_pairs = [
        (str(step["node_id"]), str(step["answer"]))
        for path in traversal_paths
        for step in path
    ]
    tutor = StudentTutor(
        llm=llm,
        nodes_by_id=nodes_by_id,
        root_id=str(result["root_id"]),
        target_leaf_id=str(result["leaf_id"]),
        target_pairs=target_pairs,
        problem_text=str(result["problem"]),
    )

    opening = (
        "Let's work through this practice problem together. "
        "Ask clarifying questions anytime. If you want to commit a decision, say it directly. "
        "If you want to revisit, say 'go back'."
    )
    first_message = f"{opening}\n\n{_render_current_step(tutor)}"
    tutor.state["response"] = first_message
    tutor.state["chat_history"].append({"role": "assistant", "content": first_message})

    st.session_state.generation_result = result
    st.session_state.tutor = tutor
    st.session_state.tutor_result = _build_tutor_result(tutor)


def _render_generation_section() -> None:
    st.header("1) Generate Practice Problem")
    
    # 1. Move the provider selectbox OUTSIDE the form so callbacks work
    provider = st.selectbox(
        "LLM provider",
        ["ollama", "openai"],
        key="llm_provider",
        on_change=_sync_model_default,
    )

    with st.expander("Optional traversal JSON", expanded=False):
        traversal_json = st.text_area(
            "Paste a single or joint traversal JSON payload",
            value="",
            height=220,
            key="traversal_json_input",
        )
        if st.button("Load traversal JSON"):
            try:
                payload = json.loads(traversal_json)
                st.session_state.loaded_traversal = _build_traversal_from_payload(payload)
                st.success("Loaded traversal JSON.")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.loaded_traversal:
        st.caption("Using loaded traversal JSON for generation.")

    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        with col1:
            model_id = st.number_input("Model ID", value=1082, step=1)
            strategy = st.selectbox("Sampling strategy", ["balanced-leaf", "all-paths"])
        
        with col2:
            # 2. Use the current session state for the model name input
            model_name = st.text_input(
                f"{provider.title()} model", 
                key="llm_model_name"
            )
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
            
            if provider == "openai":
                st.caption("Uses OPENAI_API_KEY from the environment.")

        submitted = st.form_submit_button("Generate")
        
        if submitted:
            try:
                with st.spinner("Sampling traversal and generating practice problem..."):
                    _generate_problem(
                        model_id=int(model_id),
                        strategy=str(strategy),
                        provider=str(provider),
                        model_name=model_name.strip(),
                        temperature=float(temperature),
                        traversal=st.session_state.loaded_traversal,
                    )
                st.success("Generated practice problem and initialized tutor state.")
            except Exception as exc:
                st.error(str(exc))


def _render_problem_details() -> None:
    result = st.session_state.generation_result
    if not result:
        st.info("Generate a practice problem to start tutoring.")
        return

    st.subheader("Generated Practice Problem")
    with st.container(border=True):
        st.write(result["problem"])


def _render_tutor_section() -> None:
    st.header("2) Chat Tutor")
    tutor = st.session_state.tutor
    tutor_result = st.session_state.tutor_result
    if not tutor or not tutor_result:
        st.info("Tutor will appear after generation.")
        return
    st.caption(f"Current explored path: {tutor_result['path_display']}")

    chat_window = st.container(border=True, height=420)
    with chat_window:
        for turn in tutor.state["chat_history"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

    if tutor_result["completed"]:
        st.success("Session complete.")
        return

    student_input = st.chat_input("Ask a question or make your decision...")
    if student_input:
        _ = tutor.step(student_input)
        st.session_state.tutor_result = _build_tutor_result(tutor)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="DBL Practice Tutor", layout="wide")
    st.title("DBL Practice Problem + Traversal Tutor")
    _init_state()
    _render_generation_section()
    _render_problem_details()
    _render_tutor_section()


if __name__ == "__main__":
    main()
