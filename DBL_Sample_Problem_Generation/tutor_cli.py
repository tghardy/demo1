#!/usr/bin/env python
"""Interactive CLI for the traversal tutor LangGraph."""

from __future__ import annotations

import argparse
import json

from dbl_student_rag import StudentTraversalTutor


def _load_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an interactive tutor over a generated practice problem and traversal path"
    )
    parser.add_argument(
        "--session-json",
        required=True,
        help="Path to JSON payload with keys: problem, traversal_path, root_id, leaf_id",
    )
    args = parser.parse_args()

    payload = _load_payload(args.session_json)
    tutor = StudentTraversalTutor.from_generation_result(payload)

    print(tutor.start()["response"])
    while True:
        student_input = input("\nstudent> ").strip()
        if student_input.lower() in {"quit", "exit"}:
            print("Exiting tutor session.")
            break

        result = tutor.step(student_input)
        print(result["response"])

        if result["completed"]:
            print("Session complete.")
            break


if __name__ == "__main__":
    main()
