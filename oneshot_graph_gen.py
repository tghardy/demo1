from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
from neo4j import GraphDatabase, Driver
from DBL_Sample_Problem_Generation.dbl_problem_generator import Neo4jTraversalSampler, PracticeProblemGenerator, SampleTraversal, JointTraversal
from rich import print
from rich.table import Table
from typing import Sequence
import json
from langchain_community.graphs import Neo4jGraph

class OneShotGenerator:
    def __init__(self, llm: ChatOllama, driver: Driver):
        self.llm = llm
        self.driver=driver
        self.schema = self._load_graph_data()
        print(f"DEBUG: Schema loaded: {self.schema}")
        self.sampler = Neo4jTraversalSampler.from_environment()
        self.practice_generator = PracticeProblemGenerator(self.llm, validation_llm=self.llm)

    def _load_graph_data(self) -> str:
        query = """
        MATCH (n)
        WHERE n.type <> 'NULL'
        WITH collect(DISTINCT {
            id: n.id, 
            content: n.content, 
            type: n.type
        }) AS nodes
        
        OPTIONAL MATCH (a)-[r]->(b)
        WHERE r.content <> 'null'
        WITH nodes, collect(DISTINCT CASE WHEN r IS NOT NULL THEN {
            source: a.id, 
            target: b.id, 
            content: r.content
        } ELSE null END) AS raw_rels
        
        RETURN nodes, [rel IN raw_rels WHERE rel IS NOT NULL] AS relationships
        """
        
        with self.driver.session() as session:
            record = session.run(query).single()
            
            if record and record["nodes"]:
                # Explicitly construct the dictionary in the exact order you want
                ordered_graph = {
                    "nodes": record["nodes"],
                    "relationships": record["relationships"]
                }
                # json.dumps will respect this exact top-to-bottom order
                return json.dumps(ordered_graph, indent=2)
            
        return "{}"

    def _gen_thought_process(self, prompt: str):
        p = SystemMessage(f"""You are a chatbot that is developing the initial thought process of a Decision Model.
        The user will have a request to generate some sort of graph.
        Your job is to draft an initial thought process that covers how to handle problems of this type.
        For example, if the user asks about model selection, you would write a comprehensive summary of how to decide what model to use.
        Only respond with your summary, and ensure it is readable by a future chatbot that will not see these instructions.
        Here is the user's prompt:""")
        prompt = (p, HumanMessage(prompt))
        return self.llm.invoke(prompt).content

    def _gen_json(self, prompt: str, info: str):
        """
        Generates a json knowledge graph based on a user prompt.
        """
        print("[green] Generating JSON schema...[/green]")
        instructions = SystemMessage(f"""
# Purpose
Create student-friendly Pattern Expert Process (PEP) decision models that show how an expert reasons through a topic. Build the model as a tree that can be converted into a Neo4j graph.

# General Guidelines
- Write for students who are learning the topic for the first time.
- Keep language simple, concrete, and diagnostic.
- Prefer many small, clear questions over a few vague or abstract questions.
- Build trees with three node types only: `category`, `decision_point`, and `trait`.
- Use only `HAS_CHILD` relationships between nodes. Use the 'content' attribute to store text in relationships and in nodes.
- Use only 'TreeNode' nodes, with types being stored as a node attribute.
- Make every relationship label represent the answer or branch choice a student would follow.
- Treat traits as terminal identifiers that describe characteristics of the problem.

# Tree Design Rules
- Start with a broad root `category` that represents the overall subject.
- Use `category` nodes only to group **independent dimensions** of reasoning.
- Use subcategory `category` nodes only when they group potentially related decisions that represent clearly separate reasoning dimensions within a larger idea.
- Do not place overlapping or near-duplicate decision points under the same category or subcategory.
- Do not go directly from a `category` node to a `trait` node.
- Use `decision_point` nodes for diagnostic questions that guide the student toward a clearer distinction.
- Give each `decision_point` at least two child branches when possible.
- End a branch with a `trait` as soon as a distinct characteristic has been logically isolated.
- Do not add children under `trait` nodes.

# Skill: Build a PEP Model
## Initial JSON development
1. Identify the main reasoning dimensions implied by the subject and learning objectives.
2. Group independent dimensions under category nodes.
3. Turn each dimension into simple diagnostic questions.
4. Add answer branches that are clear and easy to follow.
5. Terminate each branch with a trait when the problem characteristic is distinct.

## Validate the model
Check the finished tree for:
- No trait nodes with children.
- No direct category-to-trait jumps.
- No duplicate or overlapping decision points under the same category.
- Clear student-facing language.
- Relationship labels that read like answer choices.
- Every decision node has exactly one parent node. No overlapping.
- Content of each node is either a category name, trait name, or a question.

## Format the output
When the user asks for JSON or a graph, provide:
- A `nodes` array where every node has `id`, `content`, and `type`.
- A `relationships` array where every relationship has `source`, `target`, and `type`.

# Output Standards
- Use stable, readable IDs such as `n1`, `n2`, `n3`.
- Keep node content concise.
- Phrase decision points as questions.
- Phrase traits as noun phrases, not commands.
- Relationship labels should be clear answers to questions.
- If a requested branch would produce an action instead of a trait, rewrite it as the underlying problem characteristic.
- Everything underneath category nodes should be separate and unconnected. E.g. nodes underneath two different categories should never lead to the same node- create multiple similar nodes if necessary.
- Consolidate nodes whenever possible. If the logic of something depends on the state of a value, the answers to that node should be the various states (not 'yes' or 'no').
- Use reasoning or <scratchpad> chunks to draft your model before finalizing output.

# Example Pattern
For a statistics topic, a branch might ask: “What kind of outcome is being explained?” with branches such as “continuous,” “categorical,” and “ordered.” Each branch should lead to additional diagnostic questions or terminal traits such as “Continuous Outcome,” not actions such as “Use Linear Regression.”

Note that this is Stage 2 of this process. An expert has already gone through this problem and outlined some of their thoughts on this problem.
Use these notes as you generate the graph, but feel free to deviate if necessary.
{info}

Proceed with 1. DRAFTING a graph in your reasoning, and 2. Outputting a finalized graph json.
""")
        full_prompt = (HumanMessage(prompt), instructions)
        
        return self.llm.invoke(full_prompt).content

    def _generate_node_tool(self):
        @tool
        def generate_node(id: str, content:str, type:str):
            """
            Generate a new node in a knowledge graph.

            Args:
                id: A string representing the new node's ID
                content: A string representing the main content of the node
                type: Either 'category', 'trait', or 'decision_point'
            
            Returns a bool indicating success of the tool call
            """
            try:
                self.driver.execute_query("""
                CREATE (n:TreeNode {id: $node_id,
                content: $content,
                type: $type})
                """, node_id=id, content=content, type=type)
                return True
            except Exception as e:
                print(f"Failed to generate node: {e}")
                return False

        return generate_node 

    def _generate_rel_tool(self):
        @tool
        def generate_relation(source_id:str, target_id:str, content:str):
            """
            Generate a relationship between the source and target nodes.

            Args:
                source_id: ID of the source node (or parent node)
                target_id: ID of the target (or child) node
                content: String describing the relation

            Returns a bool based on successful execution.
            """
            try:
                self.driver.execute_query("""
                MATCH (n:TreeNode {id: $parent_id})
                MATCH (m:TreeNode {id: $child_id})
                CREATE (n)-[:HAS_CHILD {content: $content}]->(m)
                """, parent_id=source_id, child_id=target_id, content=content)

                return True
            except Exception as e:
                print(f"Failed to generate relation: {e}")
                return False

        return generate_relation

    def _json_to_graph(self, schema: str):
        generate_node = self._generate_node_tool()
        generate_relation = self._generate_rel_tool()
        llm_tools = self.llm.bind_tools([generate_node, generate_relation])

        prompt = HumanMessage(f"""
                    Generate a knowledge graph using the following specs given by a colleague. 
                    Use the generate_node and generate_relation functions in order to do this.
                    Please be exact and complete the whole graph.
                    You have one 'turn' in the conversation to do this. Ensure that you generate ALL nodes and relationships before ending your response.
                    Here is the conversation the schema is found in: {schema}
""")

        print("[green] Generating Nodes... [/green]")

        response = llm_tools.invoke([prompt])
        print(response.content)

        for tool_call in response.tool_calls:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {})

            if tool_name is None and isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

            if tool_name == "generate_node":
                generate_node.invoke(tool_args)
            elif tool_name == "generate_relation":
                generate_relation.invoke(tool_args)

            else:
                print(f"[red bold] WARNING: Tool not called: {tool_name} with args {tool_args}[/red bold]")

    def generate_graph(self, prompt: str):
        try:
            self.driver.execute_query("MATCH (n) DETACH DELETE n")
            expert_thoughts = self._gen_thought_process(prompt)
            json_schema = self._gen_json(prompt, expert_thoughts)
            self._json_to_graph(json_schema)
            print("[blue bold] Code complete! [/blue bold]")
            self.schema = json_schema

        except Exception as e:
            print(f"Code failed: {e}")
            return False

# TODO: TEST THIS FUNCTION
    def regenerate_graph(self, prompt:str):
        # Regenerate the graph in self.schema with a prompt
        try:
            message1 = SystemMessage(f"""
            {self.instructions}

            **SPECIAL INSTRUCTIONS**

            You are modifying a json schema from an AI according to the wishes of the user.
            Here is the AI's initial attempt. Keep basic schema (types, etc.) the same, but modify/add/remove nodes and relations as the user requests.
            Please respond with the full updated schema.
            """)

            message2 = AIMessage(self.schema)

            message3 = HumanMessage(prompt)

            self.schema = self.llm.invoke([message1, message2, message3]).content
            self._json_to_graph(self.schema)

        except Exception as e:
            print(f"Error regenerating graph: {e}")

    def generate_traversals(self, ids: list[str]):
        traversals = []
        for id in ids:
            traversals.append(self.sampler.sample_path_from_leaf(id))
        if len(traversals) == 1:
            return traversals[0]
        elif len(traversals) > 1:
            return JointTraversal(traversals)
        else:
            raise Exception("Error: Traversal not generated!")

    def generate_problem(self, traversals):
        return self.practice_generator.generate(traversals)

    def _generate_null_node(self):
        self.driver.execute_query("""
            MERGE (n:TreeNode {id: '-n1'})
            ON CREATE SET n.content = 'NULL', n.type = 'NULL'
            WITH n
            MATCH (m:TreeNode)
            WHERE m.id <> '-n1'
            MERGE (n)<-[:HAS_CHILD]-(m)
        """)

    def grade_response(self, answer, question, traversal_key):
        # Need to generate a traversal from the student answer
        # Assume traversal_key is the right one

        self._generate_null_node()
        @tool
        def select_path(ids: Sequence):
            """
            Generates a Traversal object to be graded. Takes in node ids in order. Returns 'N/A' if the length of the sequence is 1.
            If an id is '-n1', it is treated as a NULL node that connects both ways to every other node in the graph. This is useful to represent logical gaps or inconsistencies.

            Args:
                ids: list, contains node ids in order of student logic

            Returns: 'N/A' or a traversal object.
            """
            return ids
        
        def _select_path(ids: Sequence):

            if len(ids) == 1:
                return "N/A"
            try:
                r = self.sampler.sample_from_id_list(ids)
                return r
            except Exception as e:
                print(f"[red bold] WARNING: toolcall failed: {e}[/red bold]")
                return "N/A"

        grader = self.llm.bind_tools([select_path])

        prompt = f"""
                    You are mapping a student's response to this knowledge graph schema. 
                    Use the select_path tool to input the sequence of node id's the student most likely followed.
                    If the student has illogical responses or gaps in logic, map it to the null node.
                    Follow whatever path the student took, even if it is illogical (use the NULL node to your advantage).

                    Here is the question the student is answering: {question}

                    Here is the knowledge graph to work with: {self.schema}

                    Here is the student's answer: {answer}

                    If necessary, you may call select_path multiple times (for example, if the student went through multiple branches of a tree in their answer. If a student talks about plotting data and cleaning data, and those are both trees in the schema, please select multiple branches).
                    HOWEVER! Please make sure you are mapping the ACTUAL train of thought of the student. If they got the right answer, but didn't explain it (or had faulty reasoning), don't give them credit for it! Send it to the null node instead.
                """

        print("[green bold]Grading response...[/green bold]")
        response = grader.invoke(prompt)
        print("[green bold]Forming traversals...[/green bold]")

        traversals = []
        for tool_call in response.tool_calls:
            try:
                traversals.append(_select_path(**tool_call['args']))
            except Exception as e:
                print(f"[red bold]WARNING: tool call failed with args {tool_call['args']}")

        if len(traversals) == 1:
            traversal = traversals[0]
        elif len(traversals) > 1:
            traversal = JointTraversal(traversals)
        else:
            raise Exception("Error: not enough traversals were made!")

        print("[blue]Calculating scores...[/blue]")
        responses = self._calculate_scores(traversal, traversal_key)
        self.driver.execute_query('MATCH (n:TreeNode {id: "-n1"}) DETACH DELETE n')
        return responses

    def _calculate_scores(self, traversal, key):
        # Create a table with no borders (box=None) or keep them if you prefer
        table = Table(show_lines=False, box=None)
        
        # Add your column headers with the styling you wanted
        table.add_column("Correct Answer", style="blue bold")
        table.add_column("Your Answer", style="blue bold")
        
        # Add the multi-line text as a single row
        k = key.to_prompt_lines() 
        a = traversal.to_prompt_lines()

        print(type(k))
        print(type(a))

        if isinstance(k, list | tuple):
            k = "\n\n".join(k)
        if isinstance(a, list | tuple):
            a = "\n\n".join(a)
        
        table.add_row(k, a)
        
        print(table)
        return k,a


# '''CODE FOR TESTING BELOW'''

# from dotenv import load_dotenv
# import os

# load_dotenv()
# URI = os.getenv("NEO4J_URI")
# USERNAME = os.getenv("NEO4J_USERNAME")
# PASSWORD = os.getenv("NEO4J_PASSWORD")

# AUTH = (USERNAME, PASSWORD)

# with GraphDatabase.driver(URI, auth=AUTH) as driver:
#     # print("[blue] Deleting old graph... [/blue]")
#     # driver.execute_query("MATCH (n) DETACH DELETE n")
#     g = OneShotGenerator(ChatOllama(model="gemma4:31b", num_predict=-1), driver)
#     # g.generate_graph("Generate a PEP focused on Data Cleaning. Your primary categories should be Missingness Mechanisms, Data Wrangling, and Scaling. In addition, generate a tree focused on visualizing data.")
#     paths = [["v1", "v2", "v8", "v10", "v12"]]
#     travs = g.generate_traversals(paths)
#     problem = g.generate_problem(travs).get("problem")
#     answer = input(problem + ": ")
#     g.grade_response(answer, problem, travs)

