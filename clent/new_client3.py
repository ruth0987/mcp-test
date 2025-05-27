import asyncio
import json
import logging
import sys
import os
import traceback
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import google.generativeai as genai # This is the main import we'll use for types too
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

print(dir(genai))

# Import load_dotenv for environment variable management
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to stderr
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format='%(levelname)s:%(name)s:%(message)s')

class ResearchClient:
    def __init__(self):
        self.tools = []
        self.exit_stack = AsyncExitStack()
        self._mcp_client: Optional[ClientSession] = None
        self._stop_event = asyncio.Event()

        # Configure Gemini API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file or environment.")

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Initialize conversation history, which will be passed to generate_content
        # We start with the system instruction as the first user turn, and a model acknowledgement.
        self.conversation_history: List[genai.types.Content] = [ # Use genai.types.Content
            genai.types.Content( # Use genai.types.Content
                role="user",
                parts=[genai.types.Part(text="You are a helpful research assistant. Your primary goal is to answer user questions or fulfill requests. Only use the provided tools when an explicit external action is required to complete the request. If a query can be answered conversationally, do so. If a tool call fails, explain the error to the user.")]
            ),
            genai.types.Content( # Use genai.types.Content
                role="model",
                parts=[genai.types.Part(text="Understood. How can I assist you with your research today?")]
            )
        ]

    async def connect(self) -> bool:
        """Connects to the MCP server and lists available tools."""
        logging.info("Connecting to server...")

        try:
            logging.debug("Attempting to connect to MCP server via stdio")

            server_params = StdioServerParameters(
                command="python3",
                args=["-m", "server.tools.new_tools"],
                cwd="/Users/ruthwikreddy/my_uv_project", # Ensure CWD is set if tools.py relies on it
                env=os.environ.copy() # Pass current environment variables if needed
            )

            read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_client(server_params))

            logging.debug("Stdio connection established. Creating ClientSession...")
            self._mcp_client = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            logging.debug("Initializing MCP client...")
            await self._mcp_client.initialize()

            logging.debug("Listing tools from MCP server...")
            tools_response = await self._mcp_client.list_tools()
            self.tools = tools_response.tools
            logging.info(f"Connected to MCP server. Available tools: {[tool.name for tool in self.tools]}")
            return True

        except Exception as e:
            logging.error(f"Failed to connect to MCP server: {e}")
            logging.error(traceback.format_exc())
            raise

    async def cleanup(self):
        """Clean up resources and connections."""
        logging.info("Cleaning up resources...")
        self._stop_event.set()
        if self.exit_stack:
            await self.exit_stack.aclose()
        self._mcp_client = None
        logging.info("Cleanup complete.")

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call an MCP tool with the given arguments."""
        if not self._mcp_client:
            raise RuntimeError("Not connected to MCP server. Cannot call tool.")

        # MCP's call_tool expects args as a dict, so pass kwargs directly
        result = await self._mcp_client.call_tool(tool_name, kwargs)
        return result

    async def process_query(self, user_query: str):
        """
        Processes a user query and orchestrates multi-turn interactions
        including tool calls.
        """
        if user_query.strip():
            # Add new user query to the conversation history
            self.conversation_history.append(
                genai.types.Content(role="user", parts=[genai.types.Part(text=user_query)])
            )
            print(f"You: {user_query}") # Echo user input

        # Prepare Gemini's tools list from MCP tools
        gemini_tools = []
        for tool in self.tools:
            tool_schema = genai.types.Tool( # Use genai.types.Tool
                function_declarations=[{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {}
                }]
            )

            # --- Define Parameters for Each Specific Tool ---
            # NOTE: Ensure 'properties' and 'required' match your actual tool implementations.
            # If your tool has a 'default' value for a parameter, remove it from 'required'.
            if tool.name == "search_papers":
                tool_schema.function_declarations[0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string", "description": "The topic or keywords to search for." },
                        "max_results": { "type": "integer", "description": "Maximum number of results to return." }
                    },
                    "required": ["topic"]
                }
            elif tool.name == "extract_info":
                tool_schema.function_declarations[0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "paper_id": { "type": "string", "description": "The ID of the paper to look for." }
                    },
                    "required": ["paper_id"]
                }
            elif tool.name == "read_from_db":
                tool_schema.function_declarations[0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "conn_str": { "type": "string", "description": "The PostgreSQL connection string (e.g., 'dbname=test user=myuser password=mypass')." },
                        "query": { "type": "string", "description": "The SQL query to execute." },
                        "params": { "type": "array", "items": {"type": "string"}, "description": "Parameters for the SQL query." }
                    },
                    "required": ["conn_str", "query"]
                }
            elif tool.name == "write_to_db":
                tool_schema.function_declarations[0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "conn_str": { "type": "string", "description": "The PostgreSQL connection string (e.g., 'dbname=test user=myuser password=mypass')." },
                        "query": { "type": "string", "description": "The SQL query to execute." },
                        "params": { "type": "array", "items": {"type": "string"}, "description": "Parameters for the SQL query." }
                    },
                    "required": ["conn_str", "query"]
                }
            else:
                logging.warning(f"No specific schema defined for tool '{tool.name}'. Using empty parameters.")

            gemini_tools.append(tool_schema)

        # Loop to handle multi-turn responses from Gemini (e.g., tool chaining)
        # This loop continues until Gemini provides a final text response or indicates it's done.
        while True:
            try:
                # Send the entire conversation history and tools to the model
                response_stream = await self.model.generate_content_async(
                    contents=self.conversation_history,
                    tools=gemini_tools if gemini_tools else None,
                    stream=True,
                    # Optional: Adjust generation config
                    generation_config=genai.GenerationConfig(
                        temperature=0.1, # Keep low for consistent tool use and direct answers
                    )
                )

                # Accumulate the full response from the stream
                full_response_parts = []
                async for chunk in response_stream:
                    for part in chunk.parts:
                        full_response_parts.append(part)

                # Determine if Gemini wants to call a tool or provide text
                tool_call_part: Optional[genai.types.FunctionCall] = None
                text_response_content = ""

                for part in full_response_parts:
                    if part.function_call:
                        tool_call_part = part.function_call
                        break # Assume one function call per model turn for simplicity
                    elif part.text:
                        text_response_content += part.text

                if tool_call_part:
                    tool_name = tool_call_part.name
                    tool_args = dict(tool_call_part.args) if tool_call_part.args else {}

                    logging.info(f"Gemini requests tool call: {tool_name} with args: {tool_args}")
                    print(f"Assistant (Tool Call): Calling {tool_name} with {tool_args}")

                    # Add Gemini's tool call to history
                    self.conversation_history.append(
                        genai.types.Content(role="model", parts=[genai.types.Part(function_call=tool_call_part)])
                    )

                    tool_result_content: str = ""
                    if not self._mcp_client:
                        tool_result_content = "Error: MCP client not available for tool call."
                        logging.error(tool_result_content)
                    else:
                        try:
                            tool_result_raw = await self._mcp_client.call_tool(tool_name, tool_args)
                            tool_result_content = str(tool_result_raw)
                            logging.info(f"MCP Tool {tool_name} result: {tool_result_content}")
                            print(f"Assistant (Tool Result): {tool_result_content}")
                        except Exception as tool_e:
                            tool_result_content = f"Error executing tool '{tool_name}': {tool_e}"
                            logging.error(tool_result_content)
                            logging.error(traceback.format_exc())

                    # Add tool response to history (crucial for Gemini to process result)
                    self.conversation_history.append(
                        genai.types.Content(
                            role="tool",
                            parts=[genai.types.Part.from_function_response(
                                name=tool_name,
                                response={"content": tool_result_content}
                            )]
                        )
                    )
                    # Loop will continue to allow Gemini to process the tool result and respond

                elif text_response_content:
                    # If no tool call, and there's text content, display it and end this turn
                    self.conversation_history.append(
                        genai.types.Content(role="model", parts=[genai.types.Part(text=text_response_content)])
                    )
                    print(f"Assistant:\n{text_response_content}")
                    break # Exit loop after providing a text response

                else:
                    logging.warning("Gemini returned no text content or function call for this turn. Ending turn.")
                    # This could happen if Gemini decides no further action is needed or if it's "thinking".
                    # We should probably break to avoid an infinite loop if no explicit response comes.
                    break

            except Exception as e:
                error_message = f"Error processing query with Gemini: {e}"
                logging.error(error_message, exc_info=True)
                print(f"Assistant:\nError: {error_message}")
                # Optionally, add error to history so Gemini knows about it for future turns
                self.conversation_history.append(
                    genai.types.Content(role="model", parts=[genai.types.Part(text=f"An error occurred: {error_message}")])
                )
                break # Break on unhandled errors to avoid infinite loop

    async def run_console(self):
        """Starts a console loop for user interaction."""
        logging.info("Research Client ready. Type your queries or 'quit' to exit.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                # Only process if there's actual user input
                if user_input.strip():
                    await self.process_query(user_input)
            except EOFError: # Ctrl-D
                logging.info("EOF detected. Exiting client.")
                break
            except KeyboardInterrupt: # Ctrl-C
                logging.info("Keyboard interrupt detected. Exiting client.")
                break
            except Exception as e:
                logging.error(f"Error in console loop: {e}", exc_info=True)

async def main():
    logging.info("Starting ResearchClient main function...")
    client = ResearchClient()
    try:
        await client.connect()
        await client.run_console()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())