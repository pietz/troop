import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs an async coroutine synchronously, preserving context variables.
    
    If there's already a running loop, it uses a thread pool to avoid conflicts.
    Otherwise, it simply runs the coroutine via asyncio.run.
    """
    ctx = contextvars.copy_context()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor() as executor:
            future = executor.submit(ctx.run, asyncio.run, coro)
            return future.result()
    else:
        return ctx.run(asyncio.run, coro)


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])
