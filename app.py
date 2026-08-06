import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from oneshot_graph_gen import OneShotGenerator
from streamlit_agraph import Config, Edge, Node, agraph
import textwrap

# 1. PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# 2. Setup Secrets & LLM
load_dotenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (USERNAME, PASSWORD)

OLLAMA_HOST = "https://ollama.com"
API_KEY = os.getenv("OLLAMA_API_KEY")

llm = ChatOllama(model="glm-5.2", reasoning=True, base_url=OLLAMA_HOST, headers={
    "Authorization": f'Bearer {API_KEY}',
    "Content-Type": "application/json"
})

# 3. Cache the Database Driver globally
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(URI, auth=AUTH)

driver = get_driver()
# Initialize your generator once
g = OneShotGenerator(llm, driver)

# 5. Cache Graph Data (Prevents querying on every UI click)
@st.cache_data
def get_graph_data():
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        nodes = {}
        edges = []

        for record in result:
            n, m, r = record["n"], record["m"], record["r"]
            
            n_content = n.get("content", "")
            m_content = m.get("content", "")
            
            if n_content == "NULL" or m_content == "NULL":
                continue 

            n_id = n.get("id", str(n.element_id))
            m_id = m.get("id", str(m.element_id))

            nodes[n_id] = {
                "label": list(n.labels)[0] if n.labels else "Node",
                "properties": dict(n),
            }
            nodes[m_id] = {
                "label": list(m.labels)[0] if m.labels else "Node",
                "properties": dict(m),
            }

            edges.append((n_id, m_id, r.get("content", "")))

    return nodes, edges

def create_agraph(nodes_dict, edges_list):
    nodes, edges = [], []
    color_map = {
        "trait": "#1f78b4",
        "decision_point": "#33a02c",
        "category": "#e31a1c",
    }

    for node_id, data in nodes_dict.items():
        node_type = data["properties"].get("type", "Unknown")
        node_content = data["properties"].get("content", "No content")
        wrapped_label = textwrap.fill(node_content, width=20) 

        nodes.append(
            Node(
                id=str(node_id), 
                label=wrapped_label,
                color=color_map.get(node_type, "#999999"),
                title=f"ID: {node_id}\nContent: {node_content}",
                shape="box"
            )
        )

    for source, target, label in edges_list:
        edges.append(Edge(source=str(source), target=str(target), label=label))

    config = Config(
        width="100%", height=700, directed=True,
        layout={
            "hierarchical": {
                "enabled": True, "direction": "UD", "sortMethod": "directed",
                "nodeSpacing": 250, "treeSpacing": 300, "levelSeparation": 200, "parentCentralization": True
            }
        },
        physics={
            "enabled": True, "solver": "hierarchicalRepulsion",
            "hierarchicalRepulsion": {"nodeDistance": 250, "springLength": 100, "damping": 0.09}
        },
        edges={"smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}},
        nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True
    )
    return nodes, edges, config


# 4. Prompt Generation Input
prompt = st.text_input("Enter in instructions for graph generation...")
if st.button("Generate Graph Schema") and prompt:
    with st.spinner("Generating..."):
        g.generate_graph(prompt)
        st.success("Graph generated successfully!")
        
        # --- NEW CODE TO FIX THE DISPLAY ---
        get_graph_data.clear()  # Purge the old cached (empty) graph
        st.rerun()              # Force the app to refresh and draw the new graph



# -------------------------------------------------------------------
# State Management
# -------------------------------------------------------------------
if "current_group" not in st.session_state:
    st.session_state.current_group = []

# -------------------------------------------------------------------
# Graph Display
# -------------------------------------------------------------------
st.title("Knowledge Graph Explorer")

nodes_data, edges_data = get_graph_data()

if nodes_data:
    agraph_nodes, agraph_edges, config = create_agraph(nodes_data, edges_data)
    clicked_node_id = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

st.divider()

# -------------------------------------------------------------------
# List Builder
# -------------------------------------------------------------------
st.subheader("Problem Generation")


all_node_ids = list(nodes_data.keys()) if nodes_data else []
st.session_state.current_group = st.multiselect(
    "Enter terminal node ID(s) to generate problems from:",
    options=all_node_ids,
    default=st.session_state.current_group,
    key="multiselect_group",
)

# -------------------------------------------------------------------
# Problem Generation & Grading
# -------------------------------------------------------------------

if "generated_problem" not in st.session_state:
    st.session_state.generated_problem = None
if "traversals" not in st.session_state:
    st.session_state.traversals = None

if st.button("Generate Problem"):
    with st.spinner("Building problem..."):
        ts = g.generate_traversals(st.session_state.current_group)
        problem = g.generate_problem(ts)["problem"]
        st.session_state.generated_problem = problem
        st.session_state.traversals = ts

if st.session_state.generated_problem:
    st.markdown(st.session_state.generated_problem)
    answer = st.text_input("Enter your answer to the question...")
    
    # ADDED BUTTON HERE: Prevents re-grading on every keystroke
    if answer and st.button("Submit Answer"):
        with st.spinner("Grading response..."):
            responses = g.grade_response(
                answer, 
                st.session_state.generated_problem, 
                st.session_state.traversals
            )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Correct Answer**")
                st.markdown(responses[0])
            with col2:
                st.markdown("**Your Answer**")
                st.markdown(responses[1])