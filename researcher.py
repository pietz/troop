#!/usr/bin/env -S uv run
# /// script
# dependencies = ['mcp', 'duckduckgo_search', 'zendriver', 'trafilatura', 'googlesearch-python']
# ///
#!/usr/bin/env python3
import asyncio
from typing import Annotated
from configparser import ConfigParser

from pydantic import Field
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
import zendriver  # awesome at fetching
import trafilatura  # awesome at extracting
from googlesearch import search

mcp = FastMCP("Researcher")


@mcp.prompt()
def prompt() -> str:
    return """
    You're an AI agent with access to web tools. Use them whenever needed.
    A helpful pattern to follow is this:
    1. Use `web_search` to find potentially relevant sites online.
    2. Use `load_page` to fetch and extract content from a specific URL.
    3. Repeat any or both of these steps until you can thoroughly answer the user.
    """


@mcp.tool()
async def web_search(
    query: Annotated[str, Field(description="The search query")],
    n: Annotated[int, Field(5, description="Number of results to return")],
) -> list[dict]:
    """Execute a web search using the given query. Searches Google first, with a fallback to DuckDuckGo."""

    try:
        # Google results are better, but non-API-key access has daily limits
        results = search(query, num_results=n, advanced=True)
        return [
            {"title": r.title, "url": r.url, "description": r.description}
            for r in results
        ]
    except Exception as e:
        print(e)

    try:
        # DuckDuckGo as a solid fallback, but also has rate limits
        results = DDGS().text(query, max_results=n)
        return [
            {"title": r["title"], "url": r["href"], "description": r["body"]}
            for r in results
        ]
    except Exception as e:
        print(e)
    return "Error: Could not fetch search results."


@mcp.tool()
async def load_page(
    url: Annotated[str, Field(description="The URL to load/fetch.")],
    max_length: Annotated[
        int, Field(10000, description="Maximum number of characters to return.")
    ],
    start_index: Annotated[
        int,
        Field(
            0,
            description="Starting index of the content to return. Helpful to iteratively 'scroll' through long pages.",
        ),
    ],
    raw: Annotated[
        bool,
        Field(
            False,
            description="Return the raw HTML instead of markdown. Helpful if the content is not extracted correctly or if the HTML structure is needed.",
        ),
    ],
) -> str:
    """Fetch a URL from the internet and extract its content in markdown."""
    assert max_length > 0
    assert start_index >= 0

    try:
        config = ConfigParser()
        config["DEFAULT"] = {"DOWNLOAD_TIMEOUT": 5}
        html = trafilatura.fetch_url(url, config=config)

        if not html:
            # Advanced fallback using zendriver
            browser = await zendriver.start(headless=True)
            page = await browser.get(url)
            await page.wait_for_ready_state("complete")
            await asyncio.sleep(1)  # Increases success rate
            html = await page.get_content()
            await browser.stop()

        if raw:
            return html[start_index : start_index + max_length]

        content = trafilatura.extract(
            html,
            output_format="markdown",
            include_images=True,
            include_links=True,
        )

        if not content:
            return f"Error: Could not extract content from {url}"

        return content[start_index : start_index + max_length]

    except Exception as e:
        return f"Error loading page: {str(e)}"


if __name__ == "__main__":
    mcp.run()
