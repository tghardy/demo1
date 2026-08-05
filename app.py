import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from oneshot_graph_gen import OneShotGenerator
from streamlit_agraph import Config, Edge, Node, agraph
import textwrap

load_dotenv()
URI = st.secrets["NEO4J_URI"]
USERNAME = st.secrets["NEO4J_USERNAME"]
PASSWORD = st.secrets["NEO4J_PASSWORD"]
AUTH = (USERNAME, PASSWORD)
OLLAMA_HOST = "https://ollama.com"
API_KEY = st.secrets["OLLAMA_API_KEY"]

llm = ChatOllama(model="glm-5.2", base_url=OLLAMA_HOST, headers={
    "Authorization": f'Bearer {API_KEY}',
    "Content-Type": "application/json"
})

st.set_page_config(layout="wide")

def get_graph_data():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")

        nodes = {}
        edges = []

        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            
            # Fetch the content to check for 'NULL'
            n_content = n.get("content", "")
            m_content = m.get("content", "")
            
            # Skip this entire relationship if either the source or target node is 'NULL'
            if n_content == "NULL" or m_content == "NULL":
                continue 

            # NEW: Extract your custom 'id' attribute. 
            # We use str(n.element_id) purely as a fallback just in case a node is missing the 'id' property.
            n_id = n.get("id", str(n.element_id))
            m_id = m.get("id", str(m.element_id))

            # Store nodes using your custom ID
            nodes[n_id] = {
                "label": list(n.labels)[0] if n.labels else "Node",
                "properties": dict(n),
            }
            nodes[m_id] = {
                "label": list(m.labels)[0] if m.labels else "Node",
                "properties": dict(m),
            }

            # Map the edge using your custom IDs
            edge_content = r.get("content", "")
            edges.append((n_id, m_id, edge_content))

    driver.close()
    return nodes, edges


def create_agraph(nodes_dict, edges_list):
    nodes = []
    edges = []
    
    # Update these keys to match the actual string values of your 'type' attributes!
    color_map = {
        "trait": "#1f78b4",     # Blue
        "decision_point": "#33a02c",    # Green
        "category": "#e31a1c",     # Red
    }

    # Create Nodes
    for node_id, data in nodes_dict.items():
        node_type = data["properties"].get("type", "Unknown")
        
        # Extract content for the tooltip
        node_content = data["properties"].get("content", "No content")
        
        # NEW: Wrap the text after a certain number of characters (e.g., 20)
        wrapped_label = textwrap.fill(node_content, width=20) 
        
        nodes.append(
            Node(
                id=str(node_id), 
                label=wrapped_label, # Use the wrapped text for the visual label
                color=color_map.get(node_type, "#999999"),
                title=f"ID: {node_id}\nContent: {node_content}", # Keep full text in tooltip
                shape="box" # NEW: 'box' shape wraps cleanly around multiline text
            )
        )
        
    # Create Edges
    for source, target, label in edges_list:
        edges.append(
            Edge(
                source=str(source), 
                target=str(target), 
                label=label 
            )
        )
        
    # Hierarchical Layout Config
    config = Config(
        width="100%", 
        height=700, 
        directed=True,
        layout={
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 250, # You can increase this if boxes are still too close
                "treeSpacing": 300,
                "levelSeparation": 200,
                "parentCentralization": True
            }
        },
        physics={
            "enabled": True,
            "hierarchicalRepulsion": {
                "centralGravity": 0.0,
                "springLength": 100,
                "springConstant": 0.01,
                "nodeDistance": 250, # Adjust this along with nodeSpacing if needed
                "damping": 0.09
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {
                "iterations": 150
            }
        },
        edges={
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.4
            }
        },
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True
    )
    
    return nodes, edges, config


# -------------------------------------------------------------------
# 1. State Management for List of Lists
# -------------------------------------------------------------------
if "list_of_lists" not in st.session_state:
    st.session_state.list_of_lists = []  # Stores [[id1, id2], [id3, id4]]

if "current_group" not in st.session_state:
    st.session_state.current_group = []  # Active sublist being built


# -------------------------------------------------------------------
# 2. Graph Display
# -------------------------------------------------------------------
st.title("Knowledge Graph Explorer")

nodes_data, edges_data = get_graph_data()

if nodes_data:
    agraph_nodes, agraph_edges, config = create_graph = create_agraph(
        nodes_data, edges_data
    )

    # agraph returns clicked node ID
    clicked_node_id = agraph(
        nodes=agraph_nodes, edges=agraph_edges, config=config
    )

    if clicked_node_id:
        st.info(f"Selected Node ID from Graph: `{clicked_node_id}`")
        # Quick-add clicked node to current building group
        if st.button(f"Add `{clicked_node_id}` to Current Group"):
            if clicked_node_id not in st.session_state.current_group:
                st.session_state.current_group.append(clicked_node_id)
                st.rerun()

st.divider()

# -------------------------------------------------------------------
# 3. List of Lists Input Builder
# -------------------------------------------------------------------
st.subheader("Configure Node Groups (List of Lists)")

col1, col2 = st.columns([2, 1])

with col1:
    # Multiselect for manually picking node IDs for the active group
    all_node_ids = list(nodes_data.keys()) if nodes_data else []

    selected_nodes = st.multiselect(
        "Build current group of Node IDs:",
        options=all_node_ids,
        default=st.session_state.current_group,
        key="multiselect_group",
    )
    # Sync multiselect state
    st.session_state.current_group = selected_nodes

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("➕ Save Group", type="primary"):
            if st.session_state.current_group:
                st.session_state.list_of_lists.append(
                    list(st.session_state.current_group)
                )
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
    st.code(
        f"current_group = {st.session_state.current_group}", language="python"
    )
    st.code(
        f"list_of_lists = {st.session_state.list_of_lists}", language="python"
    )

st.divider()

# -------------------------------------------------------------------
# 4. Problem Generation
# -------------------------------------------------------------------
st.subheader("Problem Generation")

# 1. Initialize session state variables to store the generated data
if "generated_problem" not in st.session_state:
    st.session_state.generated_problem = None
if "traversals" not in st.session_state:
    st.session_state.traversals = None

# 2. Only handle the GENERATION inside the button block
if st.button("Generate Problem from Groups"):
    if not st.session_state.list_of_lists:
        st.error("Please add at least one group of node IDs above first.")
    else:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            g = OneShotGenerator(
                llm, driver
            )
            ts = g.generate_traversals(st.session_state.list_of_lists)
            problem = g.generate_problem(ts)["problem"]
            
            # Save to session state so it survives the script rerun
            st.session_state.generated_problem = problem
            st.session_state.traversals = ts

# 3. Handle the DISPLAY and GRADING outside the button block
if st.session_state.generated_problem:
    st.markdown(st.session_state.generated_problem)

    answer = st.text_input("Enter your answer to the question...")
    if answer:
        # Re-initialize the generator and driver to grade the response
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            g = OneShotGenerator(
                llm, driver
            )
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