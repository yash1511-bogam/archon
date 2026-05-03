"""MCP client + A2A protocol support.

**MCP (Model Context Protocol)** — connect to any MCP server and use its
tools as native Archon tools. The client handles discovery, invocation,
and lifecycle management.

**A2A (Agent-to-Agent Protocol)** — publish an Agent Card so other agents
can discover and delegate tasks to your Archon agents. Consume remote
agents as pipeline steps.

Both protocols follow the industry standards:
  - MCP: JSON-RPC 2.0 over stdio or HTTP (spec 2025-11-25)
  - A2A: Agent Cards at ``/.well-known/agent.json`` (Google, Apr 2025)
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from archon.tool import ToolDef

# ══════════════════════════════════════════════════════
# MCP Client
# ══════════════════════════════════════════════════════

@dataclass
class MCPTool:
    """A tool discovered from an MCP server.

    Attributes:
        name: Tool name as advertised by the server.
        description: Human-readable description.
        input_schema: JSON Schema for the tool's parameters.
        server_name: Which MCP server provides this tool.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


class MCPClient:
    """Connect to an MCP server and use its tools as Archon tools.

    Supports stdio transport (spawn a subprocess) for local MCP servers.
    HTTP transport planned for remote servers.

    Usage::

        client = MCPClient("npx @modelcontextprotocol/server-github")
        tools = client.list_tools()
        result = client.call_tool("search_repositories", {"query": "archon"})
    """

    def __init__(self, command: str, *, name: str = "mcp") -> None:
        self.command = command
        self.name = name
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0

    def connect(self) -> None:
        """Start the MCP server subprocess."""
        parts = shlex.split(self.command)
        # The command is user-supplied by design — MCP clients are explicitly
        # configured with an intended server binary to launch. ``shlex.split``
        # is used so there is no shell interpretation.
        self._process = subprocess.Popen(  # noqa: S603 — user-configured MCP server command
            parts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Send initialize request
        self._send({"method": "initialize", "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "archon", "version": "0.2.0"},
        }})
        self._recv()  # Read initialize response
        self._send({"method": "notifications/initialized"})

    def list_tools(self) -> list[MCPTool]:
        """Discover available tools from the MCP server."""
        self._send({"method": "tools/list"})
        response = self._recv()
        tools_data = response.get("result", {}).get("tools", [])
        return [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.name,
            )
            for t in tools_data
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on the MCP server and return the result."""
        self._send({"method": "tools/call", "params": {
            "name": tool_name,
            "arguments": arguments,
        }})
        response = self._recv()
        content = response.get("result", {}).get("content", [])
        # Extract text from content blocks
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(content)

    def to_archon_tools(self) -> list[ToolDef]:
        """Convert MCP tools to Archon ToolDef objects for use in agents."""
        mcp_tools = self.list_tools()
        archon_tools: list[ToolDef] = []

        for mcp_tool in mcp_tools:
            # Create a closure that captures the tool name
            def make_fn(name: str) -> Any:
                def fn(**kwargs: Any) -> str:
                    return self.call_tool(name, kwargs)
                return fn

            properties = mcp_tool.input_schema.get("properties", {})
            params = {k: {"type": v.get("type", "string")} for k, v in properties.items()}

            archon_tools.append(ToolDef(
                name=mcp_tool.name,
                description=mcp_tool.description,
                fn=make_fn(mcp_tool.name),
                parameters=params,
            ))

        return archon_tools

    def disconnect(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    # ── JSON-RPC helpers ───────────────────────────────

    def _send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the MCP server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        self._request_id += 1
        rpc = {"jsonrpc": "2.0", "id": self._request_id, **message}
        payload = json.dumps(rpc)
        self._process.stdin.write(payload + "\n")
        self._process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """Read a JSON-RPC response from the MCP server."""
        if not self._process or not self._process.stdout:
            raise RuntimeError("MCP client not connected.")

        line = self._process.stdout.readline()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {}


# ══════════════════════════════════════════════════════
# A2A Protocol — Agent Cards
# ══════════════════════════════════════════════════════

@dataclass
class AgentSkill:
    """A skill advertised in an Agent Card.

    Attributes:
        id: Unique skill identifier.
        name: Human-readable skill name.
        description: What this skill does.
        input_modes: Supported input formats (e.g., ["text/plain"]).
        output_modes: Supported output formats.
    """

    id: str
    name: str
    description: str
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])


@dataclass
class AgentCard:
    """A2A Agent Card — published at ``/.well-known/agent.json``.

    Other agents fetch this card to discover capabilities and send tasks.

    Attributes:
        name: Agent name.
        description: What this agent does.
        url: Base URL where this agent accepts A2A task requests.
        skills: List of skills this agent can perform.
        version: Agent version string.
        provider: Organization or individual providing this agent.
    """

    name: str
    description: str
    url: str
    skills: list[AgentSkill] = field(default_factory=list)
    version: str = "0.2.0"
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the A2A Agent Card JSON format."""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "provider": self.provider,
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "inputModes": s.input_modes,
                    "outputModes": s.output_modes,
                }
                for s in self.skills
            ],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Deserialize from a dictionary (e.g., fetched from a remote agent)."""
        skills = [
            AgentSkill(
                id=s["id"], name=s["name"], description=s["description"],
                input_modes=s.get("inputModes", ["text/plain"]),
                output_modes=s.get("outputModes", ["text/plain"]),
            )
            for s in data.get("skills", [])
        ]
        return cls(
            name=data["name"], description=data["description"],
            url=data["url"], skills=skills,
            version=data.get("version", ""), provider=data.get("provider", ""),
        )


async def fetch_agent_card(url: str) -> AgentCard:
    """Fetch an A2A Agent Card from a remote URL.

    Typically called on ``{base_url}/.well-known/agent.json``.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return AgentCard.from_dict(response.json())
