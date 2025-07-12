import asyncio
import json
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process user query and interact with tools via GPT-4o"""
        messages = [{"role": "user", "content": query}]

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Convert to OpenAI tool (function) format
        functions = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            } for tool in available_tools
        ]

        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                functions=functions,
                function_call="auto"
            )

            message = response.choices[0].message

            # If the assistant wants to call a function/tool
            if message.function_call:
                tool_name = message.function_call.name
                tool_args = json.loads(message.function_call.arguments)

                # Call the tool
                result = await self.session.call_tool(tool_name, tool_args)

                # Append tool call and result to message history
                messages.append(message.model_dump())  # assistant function_call
                messages.append({
                    "role": "function",
                    "name": tool_name,
                    "content": result.content
                })
            else:
                return message.content  # Final assistant response

    async def chat_loop(self):
        """Interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
