"""LangGraph workflow for generating and validating practice problems."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import (  # pyright: ignore[reportMissingImports]
    BaseChatModel,
)
from langchain_core.output_parsers import StrOutputParser  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import ChatPromptTemplate  # pyright: ignore[reportMissingImports]
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]

from .traversal_sampler import Neo4jTraversalSampler, SampleTraversal, JointTraversal


class ValidationCriterion(TypedDict):
    """Per-step validation result."""

    node_id: str
    passed: bool
    reason: str


class GenerationState(TypedDict):
    """State carried through the LangGraph workflow."""

    traversal: SampleTraversal | JointTraversal
    model_question_context: str
    traversal_context: str
    max_attempts: int
    generation_attempt: int
    generated_problem: str
    step_results: list[ValidationCriterion]
    failed_step_ids: list[str]
    fix_notes: str
    is_valid: bool


class GenerationResult(TypedDict):
    """Final response payload from generation workflow."""

    problem: str
    traversal_path: list[dict[str, str]]
    traversal_paths: list[list[dict[str, str]]]
    model_question: str
    root_id: str
    leaf_id: str
    attempts_used: int
    is_valid: bool
    validated_steps: list[ValidationCriterion]
    failed_step_ids: list[str]
    fix_notes: str


class PracticeProblemGenerator:
    """Generate and validate practice problems with LangGraph loops."""

    DEFAULT_PROBLEM_CREATION_CRITERIA: list[str] = [
        "Write one unified practice-problem statement in a single block.",
        "Anchor the problem in one concrete, realistic scenario with specific details (actors, setting, and decision context).",
        "Include concrete evidence in the prompt context (for example: small numeric summaries, observed patterns, or short findings) so the student can reason rather than guess.",
        "Use the sampled traversal context directly to shape the scenario and constraints; do not just rephrase instructions.",
        "You may receive one or more traversals. Generate problems such that they are both valid and included as part of what the student needs to answer.",
        "The logic of your generated problem should follow this traversal; we are evaluating students on how well they understand these processes. Questions should follow the end label given in the traversals.",
        "Please ensure that, if problems have multiple parts, they are entirely related. Synthesize multiple traversals into one main idea, do not make multiple questions just because there are multiple traversals.",
        "For example, a path ending in 'chemical reaction' and 'potassium' would result in a problem discussing a chemical reaction of potassium, not a question about potassium and a question about a chemical reaction.",
        "Phrase the task so the correct final answer is the leaf node value.",
        "Make the scenario realistic and specific to a real-world application while still clear and unambiguous.",
        "Avoid meta language like 'your task is to identify the leaf node value' unless naturally embedded in scenario context.",
        "Provide all data and context needed to answer each of the decision steps in the traversal.",
    ]

    def __init__(
        self,
        llm: BaseChatModel,
        validation_llm: BaseChatModel | None = None,
        max_attempts: int = 3,
        problem_creation_criteria: list[str] | None = None,
    ):
        self.llm = llm
        self.validation_llm = validation_llm or llm
        self.max_attempts = max_attempts
        self.problem_creation_criteria = (
            list(problem_creation_criteria)
            if problem_creation_criteria is not None
            else list(self.DEFAULT_PROBLEM_CREATION_CRITERIA)
        )

        self.initial_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an instructional designer. Generate one singular high-quality practice "
                    "problem based on the provided decision-path contexts. "
                    "The student-facing output must be one unified practice-problem statement only, synthesizing all decision paths. "
                    "Prioritize concrete scenario detail and avoid generic restatements of instructions.",
                ),
                (
                    "human",
                    "Problem creation criteria:\n{problem_creation_criteria}\n\n"
                    "Model-level decision question context:\n{model_question_context}\n\n"
                    "Required final answer (leaf node value(s)): {leaf_id}\n\n"
                    "Traversal path context:\n{traversal_context}\n\n"
                    "Return only the final student-facing practice problem text in one block. "
                    "Do not include headers, labels, or extra sections.",
                ),
            ]
        )
        self.regeneration_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an instructional designer. Revise the practice problem so every sampled decision step from each of the traversals given is valid. "
                    "The student-facing output must be one unified practice-problem statement only. "
                    "Prioritize concrete scenario detail and avoid generic restatements of instructions.",
                ),
                (
                    "human",
                    "Problem creation criteria:\n{problem_creation_criteria}\n\n"
                    "Model-level decision question context:\n{model_question_context}\n\n"
                    "Required final answer (leaf node value(s)): {leaf_id}\n\n"
                    "Traversal path context:\n{traversal_context}\n\n"
                    "Previous draft:\n{previous_problem}\n\n"
                    "Invalid decision-step notes (must fix):\n{fix_notes}\n\n"
                    "Return only the corrected final student-facing practice problem text in one block. "
                    "Do not include headers, labels, or extra sections.",
                ),
            ]
        )
        self.validation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You validate educational practice problems against sampled decision steps of one or more traversals of a process diagram and return strict JSON only.",
                ),
                (
                    "human",
                    "Model-level decision question context:\n{model_question_context}\n\n"
                    "Traversal path context:\n{traversal_context}\n\n"
                    "Candidate problem:\n{generated_problem}\n\n"
                    "Decision steps to validate:\n{step_specs_text}\n\n"
                    "Return JSON with this exact schema:\n"
                    "{{\n"
                    '  "all_steps_valid": true|false,\n'
                    '  "steps": [\n'
                    "    {{\"node_id\": \"<node id>\", \"passed\": true|false, \"reason\": \"<brief reason>\"}}\n"
                    "  ],\n"
                    '  "fix_notes": "<concise notes for regeneration if any step failed>"\n'
                    "}}\n"
                    "A step is valid only if the generated problem text is consistent with that step's question and selected answer. "
                    "Also fail validation if the candidate uses separate sections like Scenario, Question, Worked Solution, or Final Answer instead of one unified problem statement. "
                    "If all steps pass, fix_notes must be an empty string.",
                ),
            ]
        )

        self.graph = self._build_graph()

    def generate(
        self,
        traversal: SampleTraversal | JointTraversal,
        max_attempts: int | None = None,
    ) -> GenerationResult:
        """Run the full LangGraph generation and validation workflow."""
        state: GenerationState = {
            "traversal": traversal,
            "model_question_context": self._format_model_question(traversal),
            "traversal_context": self._format_traversal(traversal),
            "max_attempts": max_attempts if max_attempts is not None else self.max_attempts,
            "generation_attempt": 0,
            "generated_problem": "",
            "step_results": [],
            "failed_step_ids": [],
            "fix_notes": "",
            "is_valid": False,
        }
        final_state = self.graph.invoke(state)
        return self._build_result(final_state)

    def generate_from_sampler(
        self,
        sampler: Neo4jTraversalSampler,
        model_id: int = 1082,
        strategy: str = "all-paths",
        max_attempts: int | None = None,
    ) -> GenerationResult:
        """Sample a traversal from Neo4j, then run the full generation workflow."""
        if strategy == "balanced-leaf":
            traversal = sampler.sample_balanced_leaf(model_id=model_id)
        elif strategy == "all-paths":
            traversal = sampler.sample_from_all_paths(model_id=model_id)
        else:
            raise ValueError("strategy must be one of: all-paths, balanced-leaf")

        return self.generate(traversal, max_attempts=max_attempts)

    def _build_graph(self) -> Any:
        builder = StateGraph(GenerationState)
        builder.add_node("generate_problem", self._node_generate_problem)
        builder.add_node("validate_problem", self._node_validate_problem)

        builder.add_edge(START, "generate_problem")
        builder.add_edge("generate_problem", "validate_problem")
        builder.add_conditional_edges(
            "validate_problem",
            self._next_after_validation,
            {
                "regenerate": "generate_problem",
                "done": END,
            },
        )

        return builder.compile()

    def _node_generate_problem(self, state: GenerationState) -> dict[str, Any]:
        attempt = state["generation_attempt"] + 1

        if state["generation_attempt"] == 0:
            chain = self.initial_generation_prompt | self.llm | StrOutputParser()
            generated_problem = chain.invoke(
                {
                    "problem_creation_criteria": self._format_creation_criteria(),
                    "model_question_context": state["model_question_context"],
                    "traversal_context": state["traversal_context"],
                    "leaf_id": self._format_leaf_targets(state["traversal"]),
                }
            )
        else:
            chain = self.regeneration_prompt | self.llm | StrOutputParser()
            generated_problem = chain.invoke(
                {
                    "problem_creation_criteria": self._format_creation_criteria(),
                    "model_question_context": state["model_question_context"],
                    "traversal_context": state["traversal_context"],
                    "leaf_id": self._format_leaf_targets(state["traversal"]),
                    "previous_problem": state["generated_problem"],
                    "fix_notes": state["fix_notes"] or "No explicit notes provided.",
                }
            )

        return {
            "generation_attempt": attempt,
            "generated_problem": generated_problem,
        }

    def _node_validate_problem(self, state: GenerationState) -> dict[str, Any]:
        traversals = self._get_traversals(state["traversal"])

        all_step_results: list[ValidationCriterion] = []
        all_failed_step_ids: list[str] = []
        all_fix_notes: list[str] = []

        for trav_index, traversal in enumerate(traversals):
                step_specs_text = "\n".join(
                    f"{index}. node_id={step.question.node_id} | rel_type={step.rel_type or 'n/a'} | Q={step.question.question_text} | A={step.answer.answer_text}"
                    for index, step in enumerate(traversal.steps, start=1)
                )
                
                # Format context for this specific traversal
                model_question_context = self._format_model_question(traversal)
                traversal_context = self._format_traversal(traversal)
                
                chain = self.validation_prompt | self.validation_llm | StrOutputParser()
                raw_validation = chain.invoke(
                    {
                        "model_question_context": model_question_context,
                        "traversal_context": traversal_context,
                        "generated_problem": state["generated_problem"],
                        "step_specs_text": step_specs_text,
                    }
                )
                
                expected_node_ids = [step.question.node_id for step in traversal.steps]
                parsed = self._parse_validation_json(raw_validation, expected_node_ids)
                
                # Track results per traversal
                all_step_results.extend(parsed["steps"])
                
                failed_ids = [
                    result["node_id"]
                    for result in parsed["steps"]
                    if not result["passed"]
                ]
                all_failed_step_ids.extend(failed_ids)
                
                if parsed["fix_notes"]:
                    all_fix_notes.append(f"Traversal {trav_index + 1}: {parsed['fix_notes']}")
            
        # Problem is valid only if ALL steps in ALL traversals pass
        is_valid = all(result["passed"] for result in all_step_results)
        
        return {
            "step_results": all_step_results,
            "failed_step_ids": all_failed_step_ids,
            "fix_notes": " | ".join(all_fix_notes) if all_fix_notes else "",
            "is_valid": is_valid,
        }

    @staticmethod
    def _next_after_validation(state: GenerationState) -> str:
        if state["is_valid"]:
            return "done"
        if state["generation_attempt"] >= state["max_attempts"]:
            return "done"
        return "regenerate"

    def _build_result(self, state: GenerationState) -> GenerationResult:
        traversals = self._get_traversals(state["traversal"])
        traversal_paths = [self._build_traversal_path(traversal) for traversal in traversals]
        primary_traversal = traversals[0] if traversals else state["traversal"]

        return {
            "problem": state["generated_problem"],
            "traversal_path": traversal_paths[0] if traversal_paths else [],
            "traversal_paths": traversal_paths,
            "model_question": primary_traversal.model_question,
            "root_id": primary_traversal.root_id,
            "leaf_id": primary_traversal.leaf_id,
            "attempts_used": state["generation_attempt"],
            "is_valid": state["is_valid"],
            "validated_steps": state["step_results"],
            "failed_step_ids": state["failed_step_ids"],
            "fix_notes": state["fix_notes"],
        }

    @staticmethod
    def _get_traversals(traversal: SampleTraversal | JointTraversal) -> list[SampleTraversal]:
        if isinstance(traversal, JointTraversal):
            return list(traversal.traversals)
        return [traversal]

    @staticmethod
    def _build_traversal_path(traversal: SampleTraversal) -> list[dict[str, str]]:
        return [
            {
                "node_id": step.question.node_id,
                "question": step.question.question_text,
                "answer": step.answer.answer_text,
                "rel_type": step.rel_type,
            }
            for step in traversal.steps
        ]

    @staticmethod
    def _format_leaf_targets(traversal: SampleTraversal | JointTraversal) -> str:
        leaf_ids = [candidate.leaf_id for candidate in PracticeProblemGenerator._get_traversals(traversal)]
        if not leaf_ids:
            return ""
        if len(leaf_ids) == 1:
            return leaf_ids[0]
        return ", ".join(leaf_ids)

    @staticmethod
    def _parse_validation_json(
        raw: str,
        expected_node_ids: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None

        if not payload:
            return {
                "all_steps_valid": False,
                "steps": [
                    {
                        "node_id": node_id,
                        "passed": False,
                        "reason": "Validation parser failed to decode validator output.",
                    }
                    for node_id in expected_node_ids
                ],
                "fix_notes": "Return strict JSON from validator and revise content to satisfy all decision steps.",
            }

        raw_results = payload.get("steps", [])
        normalized: list[ValidationCriterion] = []
        fallback_by_node_id = {
            str(item.get("node_id", "")).strip(): item
            for item in raw_results
            if isinstance(item, dict)
        }

        for node_id in expected_node_ids:
            found = fallback_by_node_id.get(node_id)
            if found is None:
                normalized.append(
                    {
                        "node_id": node_id,
                        "passed": False,
                        "reason": "Step result missing from validator response.",
                    }
                )
                continue

            normalized.append(
                {
                    "node_id": node_id,
                    "passed": bool(found.get("passed", False)),
                    "reason": str(found.get("reason", "")).strip() or "No reason provided.",
                }
            )

        return {
            "all_steps_valid": bool(payload.get("all_steps_valid", False)),
            "steps": normalized,
            "fix_notes": str(payload.get("fix_notes", "")).strip(),
        }

    @staticmethod
    def _format_joint_traversal(traversal: JointTraversal) -> str:
        lines = []
        for idx, t in enumerate(traversal.traversals, start=1):
            lines.append(f"=== Traversal #{idx} ===")
            lines.append(PracticeProblemGenerator._format_traversal(t))

        return "\n".join(lines)

    @staticmethod
    def _format_traversal(traversal: SampleTraversal | JointTraversal) -> str:
        if isinstance(traversal, JointTraversal):
            return PracticeProblemGenerator._format_joint_traversal(traversal)

        lines = [
            f"Model question: {traversal.model_question or 'N/A'}",
            f"Root: {traversal.root_id}",
            f"Leaf: {traversal.leaf_id}",
            "Path:",
        ]
        for index, step in enumerate(traversal.steps, start=1):
            lines.append(f"{index}. Q ({step.question.type}): {step.question.question_text}")
            
            rel_type_str = step.rel_type if step.rel_type else "[MISSING TYPE]"
            
            lines.append(f"   A: {step.answer.answer_text}")
            
        # THE FIX: Explicitly print the leaf node's content at the end of the path
        final_answer = traversal.answer.answer_text if traversal.answer else "[NO LEAF CONTENT]"
        lines.append(f"FINAL TRAIT: {final_answer}")
            
        print(f"DEBUG: TRAVERSAL FORMAT:\n" + "\n".join(lines))
        return "\n".join(lines)

    @staticmethod
    def _format_model_question(traversal: SampleTraversal | JointTraversal) -> str:
        if isinstance(traversal, JointTraversal):
            questions = [
                candidate.model_question.strip()
                for candidate in traversal.traversals
                if candidate.model_question and candidate.model_question.strip()
            ]
            if questions:
                unique_questions = list(dict.fromkeys(questions))
                if len(unique_questions) == 1:
                    return unique_questions[0]
                return "\n".join(
                    f"Traversal {index}: {question}"
                    for index, question in enumerate(unique_questions, start=1)
                )
            return "No model-level question provided; rely on traversal path context only."

        question = traversal.model_question.strip()
        if question:
            return question
        return "No model-level question provided; rely on traversal path context only."

    def _format_creation_criteria(self) -> str:
        return "\n".join(
            f"{index}. {criterion}"
            for index, criterion in enumerate(self.problem_creation_criteria, start=1)
        )
