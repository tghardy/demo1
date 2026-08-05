"""Neo4j-backed sampling of complete DBL traversals."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Any, Sequence

try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional during early setup
    load_dotenv = None


@dataclass(frozen=True)
class TraversalQuestion:
    """A single question encountered during traversal."""

    node_id: str
    question_text: str
    type: str


@dataclass(frozen=True)
class TraversalAnswer:
    """The selected answer for a traversal question."""

    answer_text: str


@dataclass(frozen=True)
class TraversalStep:
    """One step in a complete traversal path."""

    question: TraversalQuestion
    answer: TraversalAnswer
    rel_type: str = ""


@dataclass(frozen=True)
class SampleTraversal:
    """A complete path from root to leaf with question/answer pairs."""

    root_id: str
    leaf_id: str
    steps: Sequence[TraversalStep]
    answer: TraversalAnswer
    model_question: str = ""

    def to_prompt_lines(self) -> list[str]:
        """Render traversal as simple question/answer lines for prompting."""
        lines: list[str] = []
        if self.model_question:
            lines.append(f"Model question: {self.model_question}")
            
        for step in self.steps:
            lines.append(f"{step.question.question_text}: {step.answer.answer_text}")
            
        if self.answer and self.answer.answer_text:
            lines.append(f"Result: {self.answer.answer_text}")
            
        return lines

@dataclass(frozen=False)
class JointTraversal:
    """Multiple traversals, used when multiple branches are relevant to a problem."""

    traversals: list[SampleTraversal]

    def to_prompt_lines(self) -> list[str]:
        """Render both traversals as question/answer lines for prompting."""
        lines: list[str] = []
        for idx, trav in enumerate(self.traversals):
            lines.append(f"=== Traversal {idx + 1} ===")
            lines.extend(trav.to_prompt_lines())
        return lines

class TraversalSampler:
    """Base sampler for project-specific graph database traversal code."""

    def sample_complete_path(self, root_id: str | None = None) -> SampleTraversal:
        """
        Return a single complete root-to-leaf traversal.

        Replace this method body with your graph database query logic.
        """
        raise NotImplementedError("Implement graph DB traversal sampling in this class.")

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> SampleTraversal:
        """Build a `SampleTraversal` from a JSON-compatible payload."""
        steps = []
        for item in payload.get("steps", []):
            steps.append(
                TraversalStep(
                    question=TraversalQuestion(
                        node_id=item["question"]["node_id"],
                        question_text=item["question"]["question_text"],
                        type=item["question"].get("type", ""),
                    ),
                    answer=TraversalAnswer(answer_text=item["answer"]["answer_text"]),
                    # ONLY look for 'type' (and 'rel_type' just in case the dataclass serialized it that way)
                    rel_type=str(
                        item.get("type", "") or item.get("rel_type", "")
                    ).strip(),
                )
            )
        # ... rest of the method remains the same


class Neo4jTraversalSampler(TraversalSampler):
    """Traversal sampler that queries a Neo4j database."""

    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import GraphDatabase  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "The 'neo4j' package is required. Install it with: pip install neo4j"
            ) from exc

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def from_environment(cls) -> "Neo4jTraversalSampler":
        """Create sampler from NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."""
        if load_dotenv is not None:
            load_dotenv()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        missing = [
            key
            for key, value in {
                "NEO4J_URI": uri,
                "NEO4J_USER": user,
                "NEO4J_PASSWORD": password,
            }.items()
            if not value
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variable(s): {missing_str}. "
                "Set these before creating Neo4jTraversalSampler."
            )

        return cls(uri=uri, user=user, password=password)

    def verify_connectivity(self) -> None:
        """Raise an error if the Neo4j driver cannot connect."""
        self.driver.verify_connectivity()

    def close(self) -> None:
        """Close Neo4j driver resources."""
        self.driver.close()

    def sample_complete_path(self, root_id: str | None = None) -> SampleTraversal:
        """
        Sample one complete path from root to leaf.

        Manual toggle point:
        - Uncomment the method you want to use.
        """
        return self.sample_from_all_paths(root_id=root_id)
        # return self.sample_balanced_leaf(root_id=root_id)

    def sample_from_all_paths(self, root_id: str | None = None) -> SampleTraversal:
        """Sample uniformly from all root-to-terminal paths."""
        query = """
        MATCH (root:TreeNode)
        WHERE NOT ()-[:HAS_CHILD]->(root)
        
        MATCH (terminal:TreeNode)-[:HAS_CHILD*0..]->(root)
        WHERE NOT (terminal)-[:HAS_CHILD]->()
        
        MATCH path = (root)-[:HAS_CHILD*0..]->(terminal)
        
        RETURN path, '' AS model_question
        """

        with self.driver.session() as session:
            records = list(session.run(query))

        if not records:
            raise ValueError("No terminal nodes found in graph.")

        selected_record = random.choice(records)
        selected_path = selected_record["path"]
        model_question = str(selected_record.get("model_question") or "").strip()
        return self._path_to_traversal(selected_path, model_question=model_question)

    def sample_balanced_leaf(self, root_id: str | None = None) -> SampleTraversal:
        """
        Sample by first picking one terminal node uniformly, then one path to it.

        This balances leaf-node selection before path sampling.
        """
        query = """
        MATCH (terminal:TreeNode)
        WHERE NOT (terminal)-[:HAS_CHILD]->()
        WITH terminal ORDER BY rand() LIMIT 1
        
        MATCH (root:TreeNode)
        WHERE NOT ()-[:HAS_CHILD]->(root)
        
        MATCH path = (root)-[:HAS_CHILD*0..]->(terminal)
        
        RETURN path, '' AS model_question
        """

        with self.driver.session() as session:
            record = session.run(query).single()

        if not record:
            raise ValueError("No terminal nodes found in graph.")

        model_question = str(record.get("model_question") or "").strip()
        return self._path_to_traversal(record["path"], model_question=model_question)

    def sample_path_from_leaf(self, leaf_id: str) -> SampleTraversal:
        """
        Given a leaf node ID, find ONE path from root to that leaf and return it as a traversal.
        
        If multiple paths exist to the leaf, samples one uniformly at random.
        """
        query = """
        MATCH (leaf:TreeNode {id: $leaf_id})
        WHERE NOT (leaf)-[:HAS_CHILD]->()
        
        MATCH (root:TreeNode)
        WHERE NOT ()-[:HAS_CHILD]->(root)
        
        MATCH path = (root)-[:HAS_CHILD*0..]->(leaf)
        
        RETURN path, '' AS model_question
        ORDER BY rand() LIMIT 1
        """
        
        with self.driver.session() as session:
            record = session.run(query, leaf_id=leaf_id).single()
        
        if not record:
            raise ValueError(f"No path found to leaf {leaf_id}. Leaf may not exist, may not be terminal, or may not have a root.")
        
        model_question = str(record.get("model_question") or "").strip()
        return self._path_to_traversal(record["path"], model_question=model_question)

    @staticmethod
    def _path_to_traversal(path: Any, model_question: str = "") -> SampleTraversal:
        nodes = path.nodes
        rels = path.relationships
        
        if not nodes:
            raise ValueError("Returned path did not contain any nodes.")
        
        steps: list[TraversalStep] = []
        for i in range(len(rels)):
            question_node = nodes[i]
            relationship = rels[i]
            
            question = TraversalQuestion(
                node_id=str(question_node.get("id", i)),
                question_text=question_node.get("content", ""),
                type=question_node.get("type", ""),
            )
            
            # STRICTLY rely on 'type'
            if isinstance(relationship, dict):
                edge_type = relationship.get("type", "")
            else:
                # Check for a property named 'type', fallback to structural Neo4j relationship type
                edge_type = relationship.get("type") or getattr(relationship, "type", "")

            rel_type = str(edge_type).strip()
            
            # THE FIX: Assign the relationship type directly as the answer text!
            answer_text = rel_type 

            answer = TraversalAnswer(answer_text=answer_text)
            steps.append(
                TraversalStep(
                    question=question,
                    answer=answer,
                    rel_type=rel_type,  
                )
            )
        
        # Extract leaf node
        leaf_node_obj = nodes[-1]
        answer = TraversalAnswer(
            # If your leaf nodes also use a different field than "content" for their answer, 
            # you may need to update this line as well (e.g., leaf_node_obj.get("name", ""))
            answer_text=str(leaf_node_obj.get("content", ""))
        )
        
        return SampleTraversal(
            root_id=str(nodes[0].get("id", "")),
            leaf_id=str(nodes[-1].get("id", "")),
            steps=steps,
            answer=answer,
            model_question=model_question,
        )

    def sample_from_id_list(self, ids: Sequence[str], model_question: str = "") -> SampleTraversal:
        """
        Build a SampleTraversal by walking the graph for an ordered list of node IDs.

        For each consecutive pair ids[i] -> ids[i+1] this:
        - looks up both nodes in the DB,
        - finds a relationship from the first to the second (if any),
        - uses the source node's content as the question text,
        - uses a relationship property (content/label/name/answer) if present as the answer text,
            otherwise falls back to the relationship type, otherwise falls back to the next node's content/id.
        - stores the relationship type in `rel_type`.

        Args:
            ids: Ordered sequence of node IDs (root first, leaf last). Must contain >= 1 element.
            model_question: Optional model-level question included in the returned SampleTraversal.

        Returns:
            SampleTraversal with steps built from live graph data.

        Raises:
            ValueError if ids is empty, if a node is not found, or if a required relationship is missing.
        """
        if not ids:
            raise ValueError("ids must contain at least one node id")

        # Helper to extract best available textual property from a node-like mapping/object
        def _node_text(node_obj, keys=("content", "question_text", "name", "label", "id")) -> str:
            if node_obj is None:
                return ""
            # node_obj from neo4j supports .get; if not, dict lookup will be attempted
            for k in keys:
                try:
                    val = node_obj.get(k)
                except Exception:
                    try:
                        val = node_obj[k]
                    except Exception:
                        val = None
                if val:
                    return str(val)
            return ""

        steps: list[TraversalStep] = []

        with self.driver.session() as session:
            # Validate that all nodes exist (single query per node to keep behavior explicit).
            nodes_cache: dict[str, Any] = {}
            for node_id in ids:
                q = "MATCH (n:TreeNode {id: $node_id}) RETURN n LIMIT 1"
                rec = session.run(q, node_id=node_id).single()
                if not rec:
                    raise ValueError(f"Node with id '{node_id}' not found in the graph.")
                nodes_cache[str(node_id)] = rec["n"]

            # For each edge between consecutive ids, fetch the relationship (if any)
            for i in range(len(ids) - 1):
                id1 = str(ids[i])
                id2 = str(ids[i + 1])

                # Prefer to return an actual relationship if present; relationship properties are checked below.
                rel_query = """
                MATCH (n1:TreeNode {id: $id1}), (n2:TreeNode {id: $id2})
                OPTIONAL MATCH (n1)-[r]->(n2)
                RETURN n1 AS n1, r AS r, n2 AS n2, type(r) AS rel_type
                LIMIT 1
                """
                rec = session.run(rel_query, id1=id1, id2=id2).single()
                if not rec:
                    raise ValueError(f"Failed to locate nodes for ids '{id1}' or '{id2}' during relationship lookup.")

                n1 = rec["n1"]
                r = rec.get("r")
                n2 = rec["n2"]

                # Build question from the source node
                question_text = _node_text(n1)
                if not question_text:
                    question_text = f"Question at node {id1}"

                question = TraversalQuestion(node_id=id1, question_text=question_text, type=str(n1.get("type", "") if n1 else ""))

                # Pick answer text from relationship properties (content/label/name/answer), then relationship type,
                # then fallback to next node content or next node id.
                answer_text = ""
                rel_type = ""
                if r:
                    # try common property names on relationship
                    for key in ("content", "label", "name", "answer", "answer_text"):
                        try:
                            val = r.get(key)
                        except Exception:
                            try:
                                val = r[key]
                            except Exception:
                                val = None
                        if val:
                            answer_text = str(val)
                            break

                    # fallback to relationship structural type
                    try:
                        rel_type = str(r.get("type") or rec.get("rel_type") or getattr(r, "type", "")).strip()
                    except Exception:
                        rel_type = str(rec.get("rel_type") or getattr(r, "type", "") or "").strip()
                    if not answer_text and rel_type:
                        answer_text = rel_type

                # final fallback: use next node content or id
                if not answer_text:
                    answer_text = _node_text(n2) or id2

                answer = TraversalAnswer(answer_text=answer_text)
                steps.append(TraversalStep(question=question, answer=answer, rel_type=str(rel_type)))

            # Build final leaf answer from the last node
            leaf_node = nodes_cache[str(ids[-1])]
            leaf_answer_text = _node_text(leaf_node, keys=("content", "answer", "name", "label", "id"))
            if not leaf_answer_text:
                leaf_answer_text = f"Leaf: {ids[-1]}"
            leaf_answer = TraversalAnswer(answer_text=leaf_answer_text)

        return SampleTraversal(
            root_id=str(ids[0]),
            leaf_id=str(ids[-1]),
            steps=steps,
            answer=leaf_answer,
            model_question=str(model_question),
        )