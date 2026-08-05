import streamlit as st
from oneshot_graph_gen import OneShotGenerator
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

AUTH = (USERNAME, PASSWORD)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # print("[blue] Deleting old graph... [/blue]")
    # driver.execute_query("MATCH (n) DETACH DELETE n")
    g = OneShotGenerator(ChatOllama(model="gemma4:31b", num_predict=-1), driver)
    prompt = st.text_input("Enter in instructions for graph generation...")
    if prompt:
        g.generate_graph(prompt)
        st.text("Graph generated successfully!")