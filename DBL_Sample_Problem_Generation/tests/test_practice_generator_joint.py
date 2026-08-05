from dbl_problem_generator.practice_generator import PracticeProblemGenerator
from dbl_problem_generator.traversal_sampler import (
    JointTraversal,
    SampleTraversal,
    TraversalAnswer,
    TraversalQuestion,
    TraversalStep,
)


def _make_traversal(root_id: str, leaf_id: str, question_text: str, answer_text: str) -> SampleTraversal:
    return SampleTraversal(
        root_id=root_id,
        leaf_id=leaf_id,
        steps=[
            TraversalStep(
                question=TraversalQuestion(node_id=f"{root_id}-q", question_text=question_text),
                answer=TraversalAnswer(answer_text=answer_text),
            )
        ],
        model_question="What should the student choose?",
    )


def test_joint_traversal_context_is_formatted_for_generation():
    generator = PracticeProblemGenerator(llm=object(), validation_llm=object())
    joint = JointTraversal(
        [
            _make_traversal("root-a", "leaf-a", "Trait A question", "Trait A answer"),
            _make_traversal("root-b", "leaf-b", "Trait B question", "Trait B answer"),
        ]
    )

    rendered = generator._format_traversal(joint)

    assert "=== Traversal #1 ===" in rendered
    assert "=== Traversal #2 ===" in rendered
    assert "Trait A question" in rendered
    assert "Trait B question" in rendered


def test_build_result_exposes_joint_traversal_paths():
    generator = PracticeProblemGenerator(llm=object(), validation_llm=object())
    traversal_a = _make_traversal("root-a", "leaf-a", "Trait A question", "Trait A answer")
    traversal_b = _make_traversal("root-b", "leaf-b", "Trait B question", "Trait B answer")
    joint = JointTraversal([traversal_a, traversal_b])

    state = {
        "traversal": joint,
        "generated_problem": "Problem",
        "step_results": [],
        "failed_step_ids": [],
        "fix_notes": "",
        "is_valid": True,
        "generation_attempt": 1,
    }

    result = generator._build_result(state)  # type: ignore[arg-type]

    assert result["root_id"] == "root-a"
    assert result["leaf_id"] == "leaf-a"
    assert len(result["traversal_paths"]) == 2
    assert result["traversal_paths"][1][0]["node_id"] == "root-b-q"
