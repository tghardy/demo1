from DBL_Sample_Problem_Generation.dbl_problem_generator import SampleTraversal, Neo4jTraversalSampler, PracticeProblemGenerator, JointTraversal
from neo4j import Driver
from langchain_ollama import ChatOllama
from langchain.tools import tool
from rich import print
import string

print("[green bold]Beginning new test...[/green bold]")
llm = ChatOllama(model="gemma4:31b")

# question = """You are a data analyst hired by a fitness coaching company in order to evaluate coach performance.
# The data you are given is a series of client records. 
# It contains the following variables: Height, Weight, Age, Subscription Length, Number of Lessons, and a proprietary Health Score.
# You notice that there are an unusual amount of missing records in the Weight data. 
# Identify the missingness mechanism Is this data likely Missing At Random, Missing Completely At Random, or Missing Not At Random?
# """

# answer = """
# This data is Missing Not At Random. This is because the missingness of it is most likely dependent on the value of weight itself. While it's possible that age or height have some correlation with it, weight by itself is a value that people may not feel comfortable sharing if it is high or low.
# """

s = Neo4jTraversalSampler.from_environment()
t = s.sample_path_from_leaf("18c0da2d-c17f-4636-809c-2cb0e3f3d409")
t2 = s.sample_path_from_leaf("5434ea87-8d05-4cb4-b815-9c37cc162914")
jt = JointTraversal([t,t2])
problemgen = PracticeProblemGenerator(llm=llm, validation_llm=llm)

question = problemgen.generate(jt)['problem']
answer = input(question + ": ")

class SimpleGrader:
    """
    A simple grader that runs through traversals and determines if students followed them correctly. Does not map student's response to anything outside of the traversals.
    """
    def grade(self, question, answer, traversals):
        understands = {}
        for t in self.traversals:
            for step in t.steps:
                q = step.question.question_text
                if step.question.type == "category":
                    continue
                p = f"""
                    You are a professor grading a student's response to a question. To do this, you will be given a question that is part of a rubric.
                    You will determine whether the student's answer follows the logic of your answer key.
                    If the student clearly understands this answer based on their response, respond with TRUE. Otherwise, respond FALSE.
                    Here is the overall question: {question}

                    Here is the student's answer to the overall question: {answer}

                    Here is the subquestion: {q}

                    Here is the correct answer to the subquestion: {step.answer.answer_text}

                    Does the student understand this answer to the subquestion? You may answer with 0, 1, or 2.
                    0: The student is obviously wrong
                    1: The student has unclear logic and it is difficult to deduce, but it appears they are approaching proficiency
                    2: The student correctly answers the question
                    After your initial answer, please explain your reasoning.

                    PLEASE NOTE: YOU ARE ONLY EVALUATING THE STUDENT ON THIS ONE SPECIFIC SUBQUESTION. Logical flaws and failures related to other areas will be dealt with in other subquestions. ONLY FOCUS ON THIS SUBQUESTION.
                    """

                response = llm.invoke(p).content 
                print(response)
                if "0" in response.lower():
                    understands[q]=0
                elif "1" in response.lower():
                    understands[q]=1
                elif "2" in response.lower():
                    understands[q]=2
                else:
                    understands[q]="Can't Tell"

        p = f"""
        You are a professor grading a student's response to a question.

        Here is the question: {question}

        Here is their answer: {answer}

        Did they get it right? Answer with TRUE or FALSE, with your explanation afterwards.
        """

        resp = llm.invoke(p).content
        print(resp)
        if "true" in resp.lower():
            understands["Final Answer"]= True
        elif "false" in resp.lower():
            understands["Final Answer"] = False
        else:
            understands["Final Answer"] = "Can't tell"

        print(understands)
        return understands

@tool
def select_answer(option: int):
    """
    Enters an answer choice into the database.

    Args:
        option: Integer representing which answer choice is to be made.

    Returns:
        Confirmation that an answer has been selected.
    """

    return option
    
class MapGrader:
    def __init__(self, driver: Driver, model: ChatOllama):
        self.driver = driver
        self.llm = model.bind_tools([select_answer])

    def _is_trait(self, id):
        result = self.driver.execute_query(query_="""MATCH (n:TreeNode {id: $id})
                                    RETURN n.type""", id=id)
        r = result.records[0]['n.type']
        if r == 'trait':
            return True
        else:
            return False



# TODO: put llms in init statements

    def _get_next_step(self, answer, question, subquestion, rel_types, answer_ids):
        option_string = ""
        for idx, option in enumerate(rel_types):
            option_string += f"{idx}: {option}"
        option_string += f"{idx+1}: None of the above"
        prompt = f"""
                You are classifying a student's response to a question. 
                Here is the overall question: {question}
                The student's answer is {answer}.

                You are currently trying to determine what the student did at this point in their logical reasoning. This is the subquestion we are analyzing:
                {subquestion}

                From the above info, what is the student's most likely chain of reasoning relative to this subquestion? You may use the select_answer() tool to place your choice.
                (Note- it is possible that the subquestion above is simply a category name. If that is the case, please select the area they are likely focusing on).

                {option_string}
                """
        response = self.llm.invoke(prompt) 

        if response.tool_calls:
            selected_index = response.tool_calls[0]['args']['option']
        else:
            raise Exception

        if selected_index >= idx+1:
            return "N/A"

        else:
            new_id = answer_ids[selected_index]
            return new_id

    def _skip_step(self, node_id, leaf_id):
            result = self.driver.execute_query(query_="""
                // 1. Replaced :Leaf and :Category labels with :TreeNode and type properties
                MATCH lineage = (leaf:TreeNode {id: $leaf_id})<-[:HAS_CHILD*1..]-(parent:TreeNode {type: 'category'})<-[:HAS_CHILD*0..]-(root:TreeNode {type: 'category'})
                WHERE NOT ()-[:HAS_CHILD]->(root)
                WITH lineage, parent, nodes(lineage) AS trunk_nodes

                MATCH (parent)-[:HAS_CHILD*0..]->(descendant:TreeNode)
                WITH trunk_nodes, collect(DISTINCT descendant) AS subtree_nodes

                UNWIND trunk_nodes AS trunk_node
                OPTIONAL MATCH (trunk_node)-[]->(other:TreeNode)
                // 2. Replaced the labels() function with property checks on the 'type' attribute
                WHERE other.type <> 'category' AND other.type <> 'trait'

                WITH trunk_nodes, subtree_nodes, collect(DISTINCT other) AS side_nodes

                WITH trunk_nodes + subtree_nodes + side_nodes AS all_nodes
                UNWIND all_nodes AS n
                WITH DISTINCT n AS valid_node
                WHERE valid_node IS NOT NULL
                
                WITH collect(valid_node) AS pruned_graph_nodes

                MATCH (n:TreeNode {id: $node_id})-[:HAS_CHILD]->(m:TreeNode)
                WHERE n IN pruned_graph_nodes AND m IN pruned_graph_nodes
                
                RETURN m.id AS next_id
            """, node_id=node_id, leaf_id=leaf_id)

            # 3. Python Safeguard: Catch empty results smoothly
            if result.records:
                return result.records[0]['next_id']
                
            raise RuntimeError(
                f"Graph Integrity Error: Node {node_id} has no valid child in the pruned graph "
                f"for leaf {leaf_id}. Verify that the path between them exists and types are correct."
            )

    def _process_node(self, question, answer, id, parent_id, traversal):
        if id == "N/A":
            return (id, parent_id)
        if self._is_trait(id):
            return id
        else:
            query = """
            MATCH r = (n:TreeNode {id: $id})-[rel:HAS_CHILD]->(a:TreeNode)
            RETURN n.content, n.type, rel.type, a.id
            """
            result = self.driver.execute_query(query, id=id)

            rel_type = []
            answer_ids = []
            for rec in result.records:
                node_content = rec['n.content']
                node_type= rec['n.type']
                rel_type.append(rec['rel.type'])
                answer_ids.append(rec['a.id'])
            if node_type == "category":
                next_id = self._skip_step(id, traversal.leaf_id)
            else:
                next_id = self._get_next_step(answer, question, node_content, rel_type, answer_ids)
            return self._process_node(question, answer, next_id, id, traversal)

    def _map_answer(self, question, answer, traversal):
        id = traversal.root_id
        return self._process_node(question, answer, id, "Root", traversal)
        

    def _calculate_score(self, final_id, t):
        number_off = 0
        result = self.driver.execute_query("""
            MATCH (studentAnswer:TreeNode {id: $final_id})
            MATCH (actualAnswer:TreeNode {id: $final_answer})
            MATCH path = shortestPath((studentAnswer)-[*]-(actualAnswer))
            RETURN length(path) as distance
            """, final_id=final_id, final_answer=t.leaf_id)
        # TODO: FIX THIS!
        rec = result.records[0]
        if rec:
            number_off += rec['distance']

        return number_off

    def grade(self, question, answer, traversals):
        final_ids = []
        number_off = 0
        if isinstance(traversals, JointTraversal):
            traversals = traversals.traversals
        for t in traversals:
            final_ids.append(self._map_answer(question, answer, t))
        for traversal, final_id in zip(traversals, final_ids):
            if final_id == "N/A":
                print("[red bold]Warning: Unable to calculate score![/red bold]")
            else:
                number_off += self._calculate_score(final_id, traversal)
        print("FINAL SCORE: ", number_off)
        return final_ids

import os
from dotenv import load_dotenv
# Load in neo4j credentials from .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOllama(model="gemma4:31b")

# Initialize connection
AUTH = (USERNAME, PASSWORD)

from neo4j import GraphDatabase
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    grader = MapGrader(driver=driver, model=llm)
    grader.grade(question, answer, jt)

# TODO: fix debug stuff for jointtraversal iteration

