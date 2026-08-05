# DBL Sample Problem Generation

This repository is now intentionally minimal so you can plug in your existing DBL graph-database code.

## Current Focus

- Sample one complete root-to-leaf traversal (questions + selected answers)
- Pass traversal context to a LangGraph generation workflow
- Generate one practice problem from that context
- Hand off problem + traversal to an interactive tutor LangGraph
- Keep a dedicated place for DBL tree loading/ingestion code

## Project Structure

```
DBL_Sample_Problem_Generation/
├── dbl_problem_generator/
│   ├── __init__.py
│   ├── traversal_sampler.py    # SampleTraversal models + sampler base class
│   ├── practice_generator.py   # LangGraph wrapper to generate practice problems
│   └── tree_loader.py          # DBL tree loading scaffold
├── dbl_student_rag/
│   ├── __init__.py
│   └── tutor_graph.py          # LangGraph tutor that moves node-to-node across target traversal
├── cli.py                      # Minimal CLI for traversal JSON -> generated problem
├── tutor_cli.py                # Interactive tutor CLI using saved generation payload
├── main.py
├── pyproject.toml
└── README.md
```

## Quick Start

1. Install dependencies:

```bash
pip install -e .
```

2. Add your existing code into:

- `dbl_problem_generator/traversal_sampler.py`
- `dbl_problem_generator/practice_generator.py`
- `dbl_problem_generator/tree_loader.py`

3. Choose the LLM backend when running the app or CLI.

- Local Ollama is the default backend.
- OpenAI is available if `OPENAI_API_KEY` is set in your environment.
- The Streamlit UI and CLI both let you pick `ollama` or `openai` and then supply the model name for that backend.

4. Run CLI with a traversal payload:

```bash
python cli.py --traversal-json path/to/traversal.json
```

Or sample directly from Neo4j and generate in one command:

```bash
python cli.py --sample-from-db --model-id 1082 --sampling-strategy all-paths --show-traversal
```

Use OpenAI instead of Ollama:

```bash
export OPENAI_API_KEY=...
python cli.py --sample-from-db --provider openai --model gpt-4o-mini
```

Use balanced leaf sampling instead:

```bash
python cli.py --sample-from-db --model-id 1082 --sampling-strategy balanced-leaf
```

Save the generated problem + traversal for tutoring handoff:

```bash
python cli.py --sample-from-db --model-id 1082 --sampling-strategy balanced-leaf --save-session-json session.json
```

Start interactive node-by-node tutoring from that saved payload:

```bash
python tutor_cli.py --session-json session.json
```

Tutor commands:

- Type an answer directly (or use `answer: <text>`)
- `back` to move to previous node
- `repeat` to re-show the current node question
- `status` to show progress

## Neo4j Credentials Setup

Use environment variables so secrets are not hardcoded in source files.

1. Create a local env file from the template:

```bash
cp .env.example .env
```

2. Fill in your real values in `.env`:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

3. Load variables in your shell before running Python code:

```bash
python -m pip install -e .
```

`Neo4jTraversalSampler.from_environment()` automatically reads `.env`.

4. Create sampler from environment in code:

```python
from dbl_problem_generator.traversal_sampler import Neo4jTraversalSampler

sampler = Neo4jTraversalSampler.from_environment()
sampler.verify_connectivity()
traversal = sampler.sample_complete_path(model_id=1082)
sampler.close()
```

Expected traversal payload shape:

```json
{
    "root_id": "root-node-id",
    "leaf_id": "leaf-node-id",
    "model_question": "Should we approve this applicant for the premium card?",
    "steps": [
        {
            "question": {
                "node_id": "q1",
                "question_text": "What topic is this learner working on?"
            },
            "answer": {
                "answer_text": "Linear equations"
            }
        }
    ]
}
```
