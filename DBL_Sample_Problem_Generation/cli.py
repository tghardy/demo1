#!/usr/bin/env python
"""CLI scaffold for traversal-driven practice generation."""

from __future__ import annotations

import argparse
import json

from dbl_problem_generator.practice_generator import PracticeProblemGenerator
from dbl_problem_generator.llm_factory import build_chat_model
from dbl_problem_generator.traversal_sampler import (
    JointTraversal,
    Neo4jTraversalSampler,
    SampleTraversal,
    TraversalSampler,
)


def _load_traversal_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_traversal_from_payload(payload: dict) -> SampleTraversal | JointTraversal:
    traversals_payload = payload.get("traversals")
    if isinstance(traversals_payload, list):
        return JointTraversal(
            [TraversalSampler.from_payload(item) for item in traversals_payload]
        )

    if isinstance(payload.get("traversal"), dict):
        return TraversalSampler.from_payload(payload["traversal"])

    return TraversalSampler.from_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one practice problem from a sampled DBL traversal"
    )
    parser.add_argument(
        "--traversal-json",
        help="Path to JSON payload containing root/leaf/steps data",
    )
    parser.add_argument(
        "--sample-from-db",
        action="store_true",
        help="Sample traversal directly from Neo4j credentials in .env",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=1082,
        help="Model id used when sampling from Neo4j (default: 1082)",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["all-paths", "balanced-leaf"],
        default="all-paths",
        help="Traversal sampling strategy when using --sample-from-db",
    )
    parser.add_argument(
        "--show-traversal",
        action="store_true",
        help="Print sampled traversal lines before generation",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat model backend to use for generation",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b",
        help="Model name for the selected backend",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--save-session-json",
        help="Optional path to save generated problem + traversal payload for tutor phase",
    )
    args = parser.parse_args()

    if not args.sample_from_db and not args.traversal_json:
        parser.error("Provide either --traversal-json or --sample-from-db")

    if args.sample_from_db:
        sampler = Neo4jTraversalSampler.from_environment()
        try:
            sampler.verify_connectivity()
            if args.sampling_strategy == "balanced-leaf":
                traversal = sampler.sample_balanced_leaf(model_id=args.model_id)
            else:
                traversal = sampler.sample_from_all_paths(model_id=args.model_id)
        finally:
            sampler.close()
    else:
        payload = _load_traversal_payload(args.traversal_json)
        traversal = _build_traversal_from_payload(payload)

    if args.show_traversal:
        print("Sampled traversal:")
        for line in traversal.to_prompt_lines():
            print(f"- {line}")
        print()

    llm = build_chat_model(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
    )
    generator = PracticeProblemGenerator(llm=llm)
    result = generator.generate(traversal)

    if args.save_session_json:
        with open(args.save_session_json, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"Saved tutor session payload to: {args.save_session_json}")
        print()

    print("Generated practice problem:")
    print(result["problem"])
    print()

    print("Validation summary:")
    print(f"- Valid: {result['is_valid']}")
    print(f"- Attempts used: {result['attempts_used']}")
    if result["failed_step_ids"]:
        print("- Failed step node_ids:")
        for node_id in result["failed_step_ids"]:
            print(f"  - {node_id}")
    else:
        print("- Failed step node_ids: none")

    if result["fix_notes"]:
        print("- Fix notes:")
        print(result["fix_notes"])

    print()
    print("Traversal path(s) (node with selected/correct answer):")
    print(f"- Root: {result['root_id']}")
    print(f"- Leaf: {result['leaf_id']}")
    traversal_paths = result.get("traversal_paths", [result["traversal_path"]])
    for path_index, path in enumerate(traversal_paths, start=1):
        print(f"Path {path_index}:")
        for index, step in enumerate(path, start=1):
            print(f"{index}. Q[{step['node_id']}]: {step['question']}")
            print(f"   A: {step['answer']}")


if __name__ == "__main__":
    main()
