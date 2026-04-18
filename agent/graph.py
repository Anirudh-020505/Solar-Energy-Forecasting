"""
LangGraph StateGraph assembly for the Grid Optimizer agent.

Pipeline
--------
START → analyze_forecast → retrieve_guidelines → battery_optimizer → generate_report
                                                                          ↓
                                                          [generation_error set?]
                                                          ├── YES → handle_llm_error → validate_output → END
                                                          └── NO  →                    validate_output → END

build_graph() compiles and returns the graph. Call once per Streamlit
process (e.g. via @st.cache_resource) and reuse across invocations.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.state import GridOptimizerState
from agent.nodes.analyze_forecast import analyze_forecast
from agent.nodes.retrieve_guidelines import retrieve_guidelines
from agent.nodes.battery_optimizer import battery_optimizer
from agent.nodes.generate_report import generate_report
from agent.nodes.handle_llm_error import handle_llm_error
from agent.nodes.validate_output import validate_output


def _route_after_generate(state: dict[str, Any]) -> str:
    """Conditional edge: go to error handler if generation failed, else validate."""
    if state.get("generation_error"):
        return "handle_llm_error"
    return "validate_output"


def build_graph() -> Any:
    """
    Compile and return the GridOptimizer StateGraph.

    Returns
    -------
    CompiledGraph — call .invoke(initial_state) to run the full pipeline.
    """
    graph = StateGraph(GridOptimizerState)

    # Register nodes
    graph.add_node("analyze_forecast", analyze_forecast)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("battery_optimizer", battery_optimizer)
    graph.add_node("generate_report", generate_report)
    graph.add_node("handle_llm_error", handle_llm_error)
    graph.add_node("validate_output", validate_output)

    # Linear edges
    graph.add_edge(START, "analyze_forecast")
    graph.add_edge("analyze_forecast", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "battery_optimizer")
    graph.add_edge("battery_optimizer", "generate_report")

    # Conditional edge after LLM call
    graph.add_conditional_edges(
        "generate_report",
        _route_after_generate,
        {
            "handle_llm_error": "handle_llm_error",
            "validate_output": "validate_output",
        },
    )

    # Error recovery rejoins the main path
    graph.add_edge("handle_llm_error", "validate_output")
    graph.add_edge("validate_output", END)

    return graph.compile()
