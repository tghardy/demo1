import json
import re
from typing import Any, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TutorState(TypedDict):
    practice_problem: str
    nodes_by_id: dict[str, Any]
    current_node_id: str
    node_history: list[str]
    answer_history: list[str]
    target_pairs: list[tuple[str, str]]
    target_leaf_id: str
    chat_history: list[dict[str, str]]
    last_student_input: str
    # Intent Logic
    intent: str
    selected_answer: str
    router_reply: str
    response: str
    completed: bool

class StudentTutor:
    def __init__(self, llm, nodes_by_id, root_id, target_leaf_id, target_pairs, problem_text):
        self.llm = llm
        self.state: TutorState = {
            "practice_problem": problem_text,
            "nodes_by_id": nodes_by_id,
            "current_node_id": root_id,
            "target_leaf_id": target_leaf_id,
            "target_pairs": target_pairs,
            "node_history": [root_id],
            "answer_history": [],
            "chat_history": [],
            "last_student_input": "",
            "intent": "chat",
            "selected_answer": "",
            "router_reply": "",
            "response": "",
            "completed": False
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(TutorState)
        builder.add_node("analyze_input", self._analyze_input)
        builder.add_node("execute_action", self._execute_action)
        builder.add_edge(START, "analyze_input")
        builder.add_edge("analyze_input", "execute_action")
        builder.add_edge("execute_action", END)
        return builder.compile()

    def _analyze_input(self, state: TutorState):
        """Context-aware intent analysis with robust JSON parsing."""
        curr_node = state["nodes_by_id"][state["current_node_id"]]
        options_text = "\n".join([f"- {o['answer_text']}" for o in curr_node["options"]]) or "None"
        
        # Format recent chat for context
        recent_turns = state["chat_history"][-5:]
        chat_context = "\n".join([f"{t['role']}: {t['content']}" for t in recent_turns])

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert tutor. Determine the student's intent.\n"
                "Intents:\n"
                "- move: Student explicitly picks an available option.\n"
                "- back: Student wants to go to the previous step.\n"
                "- chat: Student asks a question or makes a comment.\n\n"
                "CRITICAL: If the student suggests an answer that matches an option, use 'move'.\n"
                "Return ONLY strict JSON."
            )),
            ("human", (
                "CONTEXTUAL INFO:\n"
                "Problem: {prob}\n"
                "Current Question: {q}\n"
                "Available Options:\n{opts}\n\n"
                "RECENT CONVERSATION:\n{context}\n\n"
                "NEW STUDENT MESSAGE: \"{input}\"\n\n"
                "Return JSON schema:\n"
                "{{\"intent\": \"chat|move|back\", \"selection\": \"exact option text or empty\", \"reply\": \"your conversational response\"}}"
            ))
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({
                "prob": state["practice_problem"],
                "q": curr_node["question"],
                "opts": options_text,
                "context": chat_context or "No prior conversation.",
                "input": state["last_student_input"]
            })
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}
        except:
            data = {"intent": "chat", "selection": "", "reply": "I'm listening, tell me more."}

        return {
            "intent": data.get("intent", "chat") if data.get("intent") in ["chat", "move", "back"] else "chat",
            "selected_answer": data.get("selection", ""),
            "router_reply": data.get("reply", "")
        }

    def _execute_action(self, state: TutorState):
        """State Re-injection: Always attach the current node status to the reply."""
        history = list(state["node_history"])
        ans_history = list(state["answer_history"])
        curr_id = state["current_node_id"]
        nodes = state["nodes_by_id"]
        
        new_id = curr_id
        action_msg = state["router_reply"]

        # 1. Logic for Transitions
        if state["intent"] == "back":
            if len(history) > 1:
                history.pop()
                if ans_history: ans_history.pop()
                new_id = history[-1]
                action_msg = f"🔄 {action_msg}"
            else:
                action_msg = "⚠️ You are already at the beginning."
        
        elif state["intent"] == "move":
            options = nodes[curr_id].get("options", [])
            choice = state["selected_answer"].lower().strip()
            match = next((o for o in options if o["answer_text"].lower().strip() == choice), None)
            
            if match:
                new_id = match["next_node_id"]
                history.append(new_id)
                ans_history.append(match["answer_text"])
                action_msg = f"✅ {action_msg}"
            else:
                action_msg = "❓ I didn't see that option. Please pick from the list below."

        # 2. Path Validation
        is_correct = True
        for i, (t_node, t_ans) in enumerate(state["target_pairs"][:len(ans_history)]):
            if history[i] != t_node or ans_history[i].lower() != t_ans.lower():
                is_correct = False
                break

        # 3. Completion Check
        new_node = nodes[new_id]
        reached_target = (new_id == state["target_leaf_id"])
        is_leaf = len(new_node.get("options", [])) == 0
        
        status_update = ""
        completed = False
        
        if reached_target and is_correct:
            status_update = "🎉 **Goal Reached!** You successfully solved the problem."
            completed = True
        elif is_leaf:
            status_update = "❌ **End of Path.** This doesn't seem to be the correct solution. Try going 'back'."

        # 4. State Re-injection (The UI Fix)
        full_response = f"{action_msg}\n\n{status_update}\n\n"
        full_response += f"**Current Step:** {new_node['question']}\n"
        if new_node['options']:
            full_response += "**Options:**\n" + "\n".join([f"- {o['answer_text']}" for o in new_node['options']])

        # Update persistent history
        new_chat = list(state["chat_history"])
        new_chat.append({"role": "user", "content": state["last_student_input"]})
        new_chat.append({"role": "assistant", "content": full_response})

        return {
            "current_node_id": new_id,
            "node_history": history,
            "answer_history": ans_history,
            "chat_history": new_chat,
            "response": full_response,
            "completed": completed
        }

    def step(self, user_input: str):
        self.state["last_student_input"] = user_input
        self.state = self.graph.invoke(self.state)
        return self.state["response"]