import json
import inspect

from .types import AgentFunction, ChatCompletionMessageToolCall, Response, Result, Agent
from .util import debug_print

__CTX_VARS_NAME__ = "context_variables"


async def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    functions: list[AgentFunction],
    context_variables: dict,
    debug: bool,
) -> Response:
    """Handle tool calls and return a partial response with results."""
    function_map = {f.__name__: f for f in functions}
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in function_map:
            debug_print(debug, f"Tool {name} not found in function map.")
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue

        args = json.loads(tool_call.function.arguments)
        debug_print(debug, f"Processing tool call: {name} with arguments {args}")

        func = function_map[name]
        # pass context_variables to agent functions
        if __CTX_VARS_NAME__ in func.__code__.co_varnames:
            args[__CTX_VARS_NAME__] = context_variables

        raw_result = (
            await func(**args)
            if inspect.iscoroutinefunction(func)
            else func(**args)
        )

        result = Result.from_raw(raw_result)
        partial_response.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            }
        )
        partial_response.context_variables.update(result.context_variables)
        if result.agent:
            partial_response.agent = result.agent

    return partial_response

def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def functions_to_tools(functions: list[AgentFunction]) -> list[dict]:
        """Returns a list of tool definitons based on the agent's functions."""
        tools_ = []
        for f in functions:
            tool = function_to_json(f)
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        return tools_ if tools_ else None

def handle_function_result(result, debug) -> Result:
    match result:
        case Result() as result:
            return result

        case Agent() as agent:
            return Result(
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )
        case _:
            try:
                str_result = str(result)  # Try conversion first
                return Result(value=str_result)
            except Exception as e:
                error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                debug_print(debug, error_message)
                raise TypeError(
                    error_message
                ) from e  # Preserve original error chain