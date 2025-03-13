from typing import Any

from mcp import ClientSession
from mcp.types import Tool as MCPTool
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

async def mcpprompt(session: ClientSession) -> str:
    prompts = (await session.list_prompts()).prompts
    if len(prompts) != 1:
        raise ValueError("Expected exactly one prompt")
    prompt_result = await session.get_prompt(prompts[0].name)
    
    # Extract the text from prompt messages to ensure it's serializable
    if hasattr(prompt_result, 'messages') and prompt_result.messages:
        # If there's a message with text, extract it
        for msg in prompt_result.messages:
            if hasattr(msg, 'content') and hasattr(msg.content, 'text'):
                return msg.content.text
    
    # Fallback to string representation if we can't extract text
    return str(prompt_result)
    


async def mcptools(session: ClientSession) -> list[Tool]:
    tools = (await session.list_tools()).tools
    return [_initialize_tool(session, tool) for tool in tools]


def _initialize_tool(session: ClientSession, mcp_tool: MCPTool) -> Tool:
    async def prepare_tool(
        ctx: RunContext, tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        tool_def.parameters_json_schema = mcp_tool.inputSchema
        return tool_def

    async def execute_tool(**kwargs: Any) -> Any:
        return await session.call_tool(mcp_tool.name, arguments=kwargs)

    return Tool(
        execute_tool,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        takes_ctx=False,
        prepare=prepare_tool,
    )
