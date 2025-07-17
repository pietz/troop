import json
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    TextPart,
)


class MessageDisplay:
    """Handles all message display and formatting for the troop CLI"""
    
    def __init__(self, console: Console):
        self.console = console
    
    def format_tool_params(self, args: dict) -> str:
        """Format tool parameters for display in title, max 50 chars"""
        params = json.dumps(args, separators=(",", ":"))
        if len(params) > 50:
            params = params[:47] + "..."
        return params
    
    def show_tool_execution(self, tool_name: str, args: dict, result: str):
        """Display tool execution with function-style title and result content"""
        params = self.format_tool_params(args)
        title = f"{tool_name} {params}"
        
        # Truncate result if too long
        if len(result) > 500:
            result = result[:497] + "..."
        
        panel = Panel(
            result, title=f"[bold yellow]{title}[/bold yellow]", border_style="yellow"
        )
        self.console.print(panel)
        self.console.print()
    
    def show_user_message(self, message: str):
        """Display user message in a panel"""
        self.console.print()
        user_panel = Panel(message, title="User", border_style="blue")
        self.console.print(user_panel)
        self.console.print()
    
    def prompt_user_input(self) -> str:
        """Prompt for user input with formatted prefix"""
        self.console.print("[bold green]>[/bold green]", sep="", end="")
        import typer
        message = typer.prompt("", type=str, prompt_suffix="")
        self.console.print()  # Line break after user input
        return message
    
    async def stream_agent_response(self, text_content: str, agent_name: str) -> Panel:
        """Create a panel for streaming agent response"""
        return Panel(
            text_content,
            title=agent_name.capitalize(),
            border_style="blue",
        )
    
    async def stream_simple_response(self, result):
        """Stream response chunks to console without panels"""
        async for chunk in result.stream_text(delta=True):
            self.console.print(chunk, sep="", end="")
        self.console.print()
    
    async def handle_streaming_events(self, event_stream, agent_name: str):
        """Handle streaming events from agent model request"""
        text_content = ""
        panel = None
        live = None
        
        async for event in event_stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                text_content = event.part.content
                # Start streaming display
                panel = await self.stream_agent_response(text_content, agent_name)
                live = Live(panel, console=self.console, refresh_per_second=10)
                live.start()
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                text_content += event.delta.content_delta
                if panel and live:
                    panel.renderable = text_content
                    live.update(panel)
        
        # Stop the live display
        if live:
            live.stop()
            self.console.print()  # Add newline after panel
        
        return text_content
    
    async def handle_tool_events(self, tool_stream, show_tools: bool):
        """Handle tool call and result events"""
        tool_calls = {}  # Store tool calls by their ID
        
        async for event in tool_stream:
            if isinstance(event, FunctionToolCallEvent):
                # Store the tool call information
                tool_calls[event.part.tool_call_id] = {
                    "name": event.part.tool_name,
                    "args": event.part.args_as_dict(),
                }
            elif isinstance(event, FunctionToolResultEvent) and show_tools:
                # Get the corresponding tool call
                tool_info = tool_calls.get(event.tool_call_id)
                if tool_info:
                    self.show_tool_execution(
                        tool_info["name"],
                        tool_info["args"],
                        str(event.result.content),
                    )
    
    def print_dim(self, message: str):
        """Print a dimmed message"""
        self.console.print(f"[dim]{message}[/dim]")
    
    def print_error(self, message: str):
        """Print an error message"""
        self.console.print(f"[red]Error:[/red] {message}")