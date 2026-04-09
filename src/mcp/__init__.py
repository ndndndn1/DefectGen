"""Model Context Protocol server for DefectGen.

Exposes DefectGen capabilities (model loading, defect image generation,
delivery to client-specified locations) as MCP tools so any MCP-compatible
client (Claude Desktop, Claude Code, etc.) can drive the framework.
"""
from .server import build_server, main

__all__ = ["build_server", "main"]
