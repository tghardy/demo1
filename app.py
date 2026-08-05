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
URI = st.secrets["NEO4J_URI"]
USERNAME = st.secrets["NEO4J_USERNAME"]
PASSWORD = st.secrets["NEO4J_PASSWORD"]
AUTH = (USERNAME, PASSWORD)

OLLAMA_HOST = "https://ollama.com"
API_KEY = st.secrets["OLLAMA_API_KEY"]

# Note: Make sure to use a tool-compatible model here (like llama3.1)
llm = ChatOllama(model="gemma4:31b", base_url=OLLAMA_HOST, num_predict=-1, headers={
    "Authorization": f'Bearer {API_KEY}',
    "Content-Type": "application/json"
})

# 3. Cache the Database Driver globally
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(URI, auth=AUTH)

driver = get_driver()
g = OneShotGenerator(llm, driver)

# 4. Cache Graph Data
@st.cache_data
def get_graph_data():
    with driver.session() as session:
        nodes = {}
        edges = []

        # Fetch ALL nodes first (even if they have no connections)
        node_result = session.run("MATCH (n) RETURN n LIMIT 100")
        for record in node_result:
            n = record["n"]
            if n.get("content", "") == "NULL":
                continue 
            
            n_id = n.get("id", str(n.element_id))
            nodes[n_id] = {
                "label": list(n.labels)[0] if n.labels else "Node",
                "properties": dict(n),
            }

        # Fetch only the relationships
        edge_result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        for record in edge_result:
            n, m, r = record["n"], record["m"], record["r"]
            if n.get("content", "") == "NULL" or m.get("content", "") == "NULL":
                continue 
            
            n_id = n.get("id", str(n.element_id))
            m_id = m.get("id", str(m.element_id))
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

# 5. Prompt Generation Input
prompt = st.text_input("Enter in instructions for graph generation...")
if st.button("Generate Graph Schema") and prompt:
    with st.spinner("Generating..."):
        g.generate_graph(prompt)
        st.success("Graph generated successfully!")
        get_graph_data.clear() 
        st.rerun()              

# -------------------------------------------------------------------
# State Management
# -------------------------------------------------------------------
if "list_of_lists" not in st.session_state:
    st.session_state.list_of_lists = []
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

    if clicked_node_id:
        st.info(f"Selected Node ID from Graph: `{clicked_node_id}`")
        if st.button(f"Add `{clicked_node_id}` to Current Group"):
            if clicked_node_id not in st.session_state.current_group:
                st.session_state.current_group.append(clicked_node_id)
                st.rerun()
else:
    st.info("No nodes found in the database. Try generating a graph schema above!")

st.divider()

# -------------------------------------------------------------------
# List Builder
# -------------------------------------------------------------------
st.subheader("Configure Node Groups (List of Lists)")
col1, col2 = st.columns([2, 1])

with col1:
    all_node_ids = list(nodes_data.keys()) if nodes_data else []
    st.session_state.current_group = st.multiselect(
        "Build current group of Node IDs:",
        options=all_node_ids,
        default=st.session_state.current_group,
        key="multiselect_group",
    )

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("➕ Save Group", type="primary"):
            if st.session_state.current_group:
                st.session_state.list_of_lists.append(list(st.session_state.current_group))
                st.session_state.current_group = []
                st.rerun()
            else:
                st.warning("Current group is empty!")
    with btn_col2:
        if st.button("🧹 Clear Active Group"):
            st.session_state.current_group = []
            st.rerun()
    with btn_col3:
        if st.button("🗑️ Reset All Groups"):
            st.session_state.list_of_lists = []
            st.session_state.current_group = []
            st.rerun()

with col2:
    st.write("### Active Structure:")
    st.code(f"current_group = {st.session_state.current_group}", language="python")
    st.code(f"list_of_lists = {st.session_state.list_of_lists}", language="python")

st.divider()

# -------------------------------------------------------------------
# Problem Generation & Grading
# -------------------------------------------------------------------
st.subheader("Problem Generation")

if "generated_problem" not in st.session_state:
    st.session_state.generated_problem = None
if "traversals" not in st.session_state:
    st.session_state.traversals = None

if st.button("Generate Problem from Groups"):
    if not st.session_state.list_of_lists:
        st.error("Please add at least one group of node IDs above first.")
    else:
        with st.spinner("Building problem..."):
            ts = g.generate_traversals(st.session_state.list_of_lists)
            problem = g.generate_problem(ts)["problem"]
            st.session_state.generated_problem = problem
            st.session_state.traversals = ts

if st.session_state.generated_problem:
    st.markdown(st.session_state.generated_problem)
    answer = st.text_input("Enter your answer to the question...")
    
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