"""Wiring the nodes into a LangGraph StateGraph.

    START
      |
    analyze_outfit ──(not clothing)──> not_an_outfit ──> END
      |
    agent  <────────────┐
      |  (tool calls?)  │
      ├──yes──> tools ──┘          (capped at settings.max_agent_steps)
      |
      no
      |
    collect_candidates -> rank -> write_up -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

try:                                   # LangGraph 1.x
    from langgraph.prebuilt import ToolNode
except ImportError:                    # newer layouts re-export it from langchain
    from langchain.tools import ToolNode  # type: ignore

from ..rag.catalog_index import CatalogIndex
from .nodes import Nodes
from .state import AgentState
from .tools import make_tools


def build_graph(llm, index: CatalogIndex, kb=None, encode_image=None):
    tools = make_tools(index, kb)
    n = Nodes(llm, tools, index, encode_image=encode_image)

    g = StateGraph(AgentState)
    g.add_node("analyze_outfit", n.analyze_outfit)
    g.add_node("not_an_outfit", n.not_an_outfit)
    g.add_node("agent", n.agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("collect_candidates", n.collect_candidates)
    g.add_node("rank", n.rank)
    g.add_node("write_up", n.write_up)

    g.add_edge(START, "analyze_outfit")
    g.add_conditional_edges("analyze_outfit", n.route_after_analyze,
                            {"agent": "agent", "not_an_outfit": "not_an_outfit"})
    g.add_edge("not_an_outfit", END)
    g.add_conditional_edges("agent", n.route_after_agent,
                            {"tools": "tools", "collect_candidates": "collect_candidates"})
    g.add_edge("tools", "agent")
    g.add_edge("collect_candidates", "rank")
    g.add_edge("rank", "write_up")
    g.add_edge("write_up", END)
    return g.compile()
