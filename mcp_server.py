from typing import Any
from mcp.server.fastmcp import FastMCP
from utils import *

mcp = FastMCP("Diaction_agent")

def format_response(response: str) -> str:
    return f"Response: {response}"

@mcp.tool()
async def run_mcp_test(text: str) -> str:
    result = """TEST"""+text 
    return format_response(result)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')