from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from langchain.tools import tool
import uuid
from langchain_ollama import ChatOllama
from rich import print
from legacy_code.breadcrumb_utils import build_breadcrumb_string, escape_rich_markup

# Load in neo4j credentials from .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOllama(model="gemma4:31b")

# Initialize connection
AUTH = (USERNAME, PASSWORD)
# Function to write nodes to neo4j
def write_child_node(driver, parent_id: str, new_node: dict):
    child_id = str(uuid.uuid4())
    driver.execute_query(
        """
        MATCH (parent:TreeNode {id: $parent_id})
        CREATE (child:TreeNode {id: $child_id, content: $content, type: $type})
        CREATE (parent)-[:HAS_CHILD {type: $rel_type}]->(child)
        """,
        parent_id=parent_id,
        child_id=child_id,
        content=new_node.get("content", "Unknown"),
        type=new_node.get("type", "Concept"),
        rel_type=new_node.get("rel_type", "Other")
    )
    return child_id 

def get_parents(driver, child_id):
    """
    Returns a list of all parent nodes as dictionaries.
    """
    cypher_query = """
    MATCH path = (root:TreeNode)-[:HAS_CHILD*0..]->(child:TreeNode {id: $child_id})
    WHERE NOT ()-[:HAS_CHILD]->(root)
    RETURN [node in nodes(path) | {
    id: node.id,
    content: node.content,
    type: node.type
    }] AS ancestors,
    [r in relationships(path) | {rel_type: r.type}] AS path_rels
    """

    result = driver.execute_query(
        cypher_query,
        child_id = child_id
    )

    if not result.records:
        return {"nodes": [], "relationships": []}
    
    record = result.records[0]
    return {
        "nodes": record["ancestors"],
        "relationships": record['path_rels']
    }

def write_prompt(driver, id, topic, current_content, max_depth, subtopics, learning_outcome):
    parents = get_parents(driver, id)

    nodes = parents['nodes']
    rels = parents['relationships']

    if not nodes:
        breadcrumb_string = "Root (No lineage found)"
    else:
        breadcrumb_string = build_breadcrumb_string(nodes, rels)
        print(f"DEBUG: NEXT NODE TYPE: {nodes[1].get('type', 'unknown') if len(nodes) > 1 else 'n/a'}")
        print(f"DEBUG: NEXT REL TYPE: {rels[0].get('rel_type', 'RELATED_TO') if rels else 'n/a'}")
        print(f"DEBUG: Breadcrumb string: {escape_rich_markup(breadcrumb_string)}")
    
    subtop_str = "" 
    for t in subtopics:
        subtop_str += t + ","


    prompt = f"""
You are developing a Pattern Expert Process (PEP) decision model for {topic}. This model is being broken up into the following subtopics: {subtopics}. 
A PEP uses decision models to structure an expert's chain of thought to diagnose how to approach a problem. We are interested in using them to help students learn how to reason like an expert.

Your role is to expand our existing Decision Model by generating the next single layer of nodes. To do this, we will give you the tree so far.
You will then add 1 or more nodes below the current node using the create_child_node function.

The graph is structured as a tree, where the top of the tree represents a broad category and is then split into smaller and smaller ideas. There are three types of nodes you may use:
- `category`: This represents a group of INDEPENDENT ideas. It may contain several decision points. These decision points must have unique and independent ideas from each other--they should be exploring completely different ideas. Students would likely consider multiple trees descending from a category node at the same time, so make sure this is possible! These may have 1 or more nodes beneath them.
- `decision_point`: This represents a diagnostic question of some sort. A student coming to this node would determine how their problem relates to this node, then choose an appropriate answer to move on through the graph. These should generally have 2+ nodes beneath them.
- `trait`: This is covered below.

CRITICAL CONCEPT: TRAITS
The overall goal of your expansion is to break up concepts into finer and finer diagnostic questions or categories until a clear **trait** can be identified. 
* A 'trait' is an identifier that defines a characteristic of the problem (e.g., 'Classification problem', 'Missing at Random', 'Continuous Data'). 
* Traits are NOT actions. Do not generate nodes that tell the user what to do (e.g., 'Use Random Forest', 'Impute with Mean' are actions and are strictly forbidden). 
* Trait nodes are terminal nodes. If further clarification is required to distinguish between different traits, use decision nodes instead.

You should generate several decision nodes as appropriate to indicate the thought process that leads to each trait node.
This is a diagnostic tool, so the thought process should be clear (i.e. don't jump right to a trait node- insert decision points until it is clear what trait leads from them).
Do not go straight from category nodes to trait nodes.

You will find more instruction on node type and relations in the documentation for create_child_node.
Relationships between nodes can be 'yes' and 'no', but may very depending on the answer to the decision point.

Note that the first layer of concepts may be very broad - it is your job to determine how to break these up using the questions an expert would ask.
You may find it helpful to generate subcategories for extremely broad tasks, ensuring that subcategories represent independent and unique dimensions of the problem an expert would evaluate.

Whenever a distinct characteristic is logically isolated, you must terminate that branch by generating a 'trait' node to limit the size of the finished tree.

Here are the appropriate learning objectives that this graph should help students to understand (if any): {learning_outcome}. (end objectives). Please remember that this graph needs to be interpretable for a student just learning about this subject. Avoid intense jargon and keep concepts and heuristics simple.

Please remember- the decision points underneath category nodes should be COMPLETELY SEPARATE IDEAS from each other. If the questions are similar, they can be consolidated into one decision path.
Good example of this: Splitting up a 'Data Modeling' category into a single question that asks whether the data is continuous, categorical, or ordinal.
Bad example of this: Splitting up a 'Data Modeling' category into three decision points, each asking if the data is continuous, if it is categorical, or if it is ordinal.
Generally, you should use decision points to help students determine where they need to go with the problem, but category nodes are helpful to break up multifaceted problems.

Remember- to help students understand complex ideas, consider breaking them up into smaller subproblems. A high school student learning this concept for the first time should understand what's going on and see a clear process (i.e. minimize 'under the hood' reasoning).
It is okay to have lots of simple nodes, as long as they help student figure out what may be unfamiliar concepts.

Try not to stray too far from the current line you are on. Other topics will be covered in other branches- focus just on the current line, without introducing new concepts.

Here is the current line we are working on right now. Please generate logical sub-nodes for the last item: {breadcrumb_string} -> {current_content}
Note that branches may be at most {max_depth} nodes deep.
"""
    
    return prompt

def llm_query(driver, query):
    result = driver.execute_query(query)

    if not result.records:
        return "No records returned."

    formatted_rows = []
    for record in result.records:
        formatted_rows.append(dict(record))

    return str(formatted_rows)

def process_node(driver, current_id, current_content, topic, depth, subtopics, max_depth=7, learning_outcome=""):
    """
    Recursive function that prompts an LLM to generate nodes for each node in the graph. Depth-first.
    """
    created_nodes = []
    @tool
    def create_child_node(node_type: str, relationship_type: str, node_content: str):
        """
        Creates a child node beneath the current parent node.

        Args:
            node_type: MUST be one of:
                - 'decision_point': A question or decision to be made
                - 'category': A conceptual bucket/grouping for related sub-trees that don't depend on a strict yes/no answer
                - 'trait': An identifying characteristic of a problem
            relationship_type: Generally either 'yes' or 'no', or descriptive labels like 'subcategory' or 'alternative' for categories.
            node_content: The text that the node will contain.
        
        Returns: id for the newly created child node.
        """
        print("DEBUG: LLM CREATING NODE")
        new_id = write_child_node(driver, parent_id=current_id, new_node={"content": node_content, "type": node_type, "rel_type": relationship_type})
        created_nodes.append((new_id, node_type, node_content))
        return new_id
    
    @tool
    def query_graph(query):
        """
        Executes a Cypher query against the current Neo4j graph.

        Use this tool when you need to inspect existing nodes, relationships, or lineage before adding new nodes.
        The graph schema is:
        - Node label: TreeNode
        - Node properties: id, content, type
        - Relationship: :HAS_CHILD {type}

        Queries must be written in Cypher.
        """
        try:
            return llm_query(driver, query)
        except Exception as e:
            return f"Query not completed: {e}"

    print("DEBUG: QUERYING LLM")
    t_llm = llm.bind_tools([create_child_node])
    p = write_prompt(driver=driver, current_content=current_content, id=current_id, topic=topic, max_depth=max_depth, subtopics=subtopics, learning_outcome=learning_outcome)

    response = t_llm.invoke(p)
    print(response.content)

    for tool_call in response.tool_calls:
        tool_name = getattr(tool_call, "name", None)
        tool_args = getattr(tool_call, "args", {})

        if tool_name is None and isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

        if tool_name == "create_child_node":
            create_child_node.invoke(tool_args)
        elif tool_name == "query_graph":
            result = query_graph.invoke(tool_args)
            print(f"DEBUG: QUERY RESULT: {result}")

    for n in created_nodes:
        if n[1] in ["decision_point", "category"] and depth < max_depth:
            process_node(driver, n[0], n[2], topic, depth=depth+1, subtopics=subtopics, max_depth=max_depth)

def make_roots(driver, topic, starting_points: list, learning_outcome: str, max_depth=7):
    for p in starting_points:
        starting_id = str(uuid.uuid4())
        print(f"[bold green]DEBUG: Processing {p} nodes[/bold green]")

        driver.execute_query(
            """
            CREATE (n:TreeNode {id: $node_id,
            content: $content,
            type: 'category'})
            """,
            node_id = starting_id,
            content=p,
        )

        process_node(driver=driver,
                        current_id=starting_id,
                        current_content=p,
                        topic=topic,
                        depth=0,
                        subtopics=starting_points, max_depth=max_depth, learning_outcome=learning_outcome)
        

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    print("DEBUG: Clearing current graph...")
    driver.execute_query("MATCH (n) DETACH DELETE n")
    lo = "Students should understand how to distinguish between MAR, MNAR, and MCAR data. Students should learn when to use various types of charts to represent data."
    make_roots(driver, "Intro to Data Cleaning", ["Data Missingness", "NA Data Strategies"], lo, max_depth=10)

        




        

