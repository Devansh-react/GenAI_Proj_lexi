"""Graph wiring only. Branch functions are pure reads of state."""
from langgraph.graph import END, START, StateGraph
from app.graph.state import QAState
from app.graph.nodes import Nodes


def recursion_limit_for(max_attempts: int) -> int:
    """Return the minimum safe cap for this graph's bounded paths.

    Each retrieval attempt has three nodes: retrieve, assemble_context, and
    evaluate_retrieval. The longest terminal path then adds one refusal node.
    One additional step is retained as the structural safety margin.
    """
    return (max_attempts * 3) + 2

def build_graph(nodes: Nodes):
    graph = StateGraph(QAState)
    graph.add_node("retrieve", nodes.retrieve); graph.add_node("assemble_context", nodes.assemble_context)
    graph.add_node("evaluate_retrieval", nodes.evaluate_retrieval); graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("format_citations", nodes.format_citations); graph.add_node("insufficient_evidence", nodes.insufficient_evidence)
    graph.add_node("error_handler", nodes.error_handler)
    graph.add_edge(START, "retrieve"); graph.add_edge("retrieve", "assemble_context"); graph.add_edge("assemble_context", "evaluate_retrieval")
    def after_evaluation(state: QAState) -> str:
        if state.get("status") == "error": return "error_handler"
        if state.get("retrieval_verdict") == "sufficient": return "generate_answer"
        return "retrieve" if state.get("attempt_count", 0) < state.get("max_attempts", 2) else "insufficient_evidence"
    graph.add_conditional_edges("evaluate_retrieval", after_evaluation)
    def after_generation(state: QAState) -> str: return "error_handler" if state.get("status") == "error" else "format_citations"
    graph.add_conditional_edges("generate_answer", after_generation)
    graph.add_edge("format_citations", END); graph.add_edge("insufficient_evidence", END); graph.add_edge("error_handler", END)
    return graph.compile()
