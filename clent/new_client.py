import asyncio
import json
import logging
import sys
import os
import traceback
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

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

        # --- MODIFIED: Initialize chat history with system instructions ---
        # Gemini does not support a "system" role directly in start_chat history.
        # Instead, we provide the initial instructions as the very first user message,
        # followed by an immediate model response to set the stage.
        self.chat = self.model.start_chat(history=[
            {
                "role": "user",
                "parts": [{"text": "You are a helpful research assistant. Your primary goal is to answer user questions or fulfill requests. Only use the provided tools when an explicit external action is required to complete the request. If a query can be answered conversationally, do so. If a tool call fails, explain the error to the user. Do you understand?"}]
            },
            {
                "role": "model",
                "parts": [{"text": "Yes, I understand. How can I help you today?"}]
            }
        ])
        self.conversation_history: List[Dict[str, Any]] = [] # To store history for client-side logging/display

    async def connect(self) -> bool:
        """Connects to the MCP server and lists available tools."""
        logging.info("Connecting to server...")

        try:
            logging.debug("Attempting to connect to MCP server via stdio")

            # Create server parameters with command and args
            server_params = StdioServerParameters(
                command="python3",
                args=["-m", "server.tools.new_tools"],
                cwd="/Users/ruthwikreddy/my_uv_project", # Ensure CWD is set if tools.py relies on it
                env=os.environ.copy() # Pass current environment variables if needed
            )

            # Use stdio_client which will handle process spawning and give us streams
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
            logging.error(traceback.format_exc()) # Log full traceback for debugging
            raise # Re-raise the exception to stop execution if connection fails

    async def cleanup(self):
        """Clean up resources and connections."""
        logging.info("Cleaning up resources...")
        self._stop_event.set() # Signal any long-running tasks to stop
        if self.exit_stack:
            await self.exit_stack.aclose() # Properly close all entered async contexts
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
        Processes user query using Gemini and available tools, maintaining conversation history.
        This method handles sending user input to Gemini, interpreting tool calls,
        executing them via MCP, and feeding results back to Gemini.
        """
        if user_query.strip():
            # Add user query to conversation history if it's a new turn from user
            self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
            # Also add to Gemini's chat history for it to maintain context
            self.chat.history.append({"role": "user", "parts": [{"text": user_query}]})

        # Prepare Gemini's tools list based on tools fetched from MCP server
        gemini_tools = []
        for tool in self.tools:
            # Define tool schemas for Gemini. These MUST match your tool implementations.
            tool_schema = {
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {} # Initialize empty, then populate below
                }]
            }

            # --- Define Parameters for Each Specific Tool ---
            if tool.name == "search_papers":
                tool_schema["function_declarations"][0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic or keywords to search for."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return."
                        }
                    },
                    "required": ["topic"]
                }
            elif tool.name == "extract_info":
                tool_schema["function_declarations"][0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string",
                            "description": "The ID of the paper to look for."
                        }
                    },
                    "required": ["paper_id"]
                }
            elif tool.name == "read_from_db":
                tool_schema["function_declarations"][0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "conn_str": {
                            "type": "string",
                            "description": "The PostgreSQL connection string (e.g., 'dbname=test user=myuser password=mypass')."
                        },
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute."
                        },
                        "params": {
                            "type": "array",
                            "items": {"type": "string"}, # Assuming simple parameters; adjust if complex types are used
                            "description": "Parameters for the SQL query."
                        }
                    },
                    "required": ["conn_str", "query"] # 'params' is optional
                }
            elif tool.name == "write_to_db":
                tool_schema["function_declarations"][0]["parameters"] = {
                    "type": "object",
                    "properties": {
                        "conn_str": {
                            "type": "string",
                            "description": "The PostgreSQL connection string (e.g., 'dbname=test user=myuser password=mypass')."
                        },
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute."
                        },
                        "params": {
                            "type": "array",
                            "items": {"type": "string"}, # Assuming simple parameters; adjust if complex types are used
                            "description": "Parameters for the SQL query."
                        }
                    },
                    "required": ["conn_str", "query"] # 'params' is optional
                }
            else:
                logging.warning(f"No specific schema defined for tool '{tool.name}'. Using empty parameters.")

            gemini_tools.append(tool_schema)

        try:
            # Send message to Gemini. If `user_query` is empty, it means we're
            # letting Gemini respond after a tool call.
            response_generator = await self.chat.send_message_async(
                user_query if user_query.strip() else "", # Send user query, or empty string to trigger continuation
                tools=gemini_tools if gemini_tools else None,
                stream=True
            )

            # Accumulate full response first
            full_response_parts = []
            async for chunk in response_generator:
                for part in chunk.parts:
                    full_response_parts.append(part)

            # Now, process the accumulated parts to determine what Gemini wants
            tool_call_part = None
            text_response_content = ""

            for part in full_response_parts:
                if hasattr(part, 'function_call') and part.function_call:
                    tool_call_part = part.function_call
                    break # Assuming only one function call per model response in most cases
                elif hasattr(part, 'text') and part.text:
                    text_response_content += part.text

            if tool_call_part:
                tool_name = tool_call_part.name
                tool_args = dict(tool_call_part.args) if tool_call_part.args else {}

                logging.info(f"Gemini requests tool call: {tool_name} with args: {tool_args}")

                # Add Gemini's tool call to history (for Gemini's context)
                self.chat.history.append({
                    "role": "model",
                    "parts": [{"function_call": {"name": tool_name, "args": tool_args}}]
                })
                # Add Gemini's tool call to client's display history
                self.conversation_history.append({
                    "role": "model",
                    "parts": [{"text": f"Calling tool: {tool_name}({tool_args})", "tool_call": tool_args}]
                })

                if not self._mcp_client:
                    error_msg = "MCP client not available for tool call. Cannot proceed."
                    logging.error(error_msg)
                    tool_result = f"Error: {error_msg}" # Prepare error for tool response
                else:
                    try:
                        tool_result_raw = await self._mcp_client.call_tool(tool_name, tool_args)
                        tool_result = str(tool_result_raw) # Ensure result is a string
                        logging.info(f"MCP Tool {tool_name} result: {tool_result}")
                    except Exception as tool_e:
                        tool_result = f"Error executing tool '{tool_name}': {tool_e}"
                        logging.error(f"Error executing tool '{tool_name}': {tool_e}")
                        logging.error(traceback.format_exc())

                # Add tool response to Gemini's chat history (Crucial for Gemini to process result)
                self.chat.history.append({
                    "role": "tool",
                    "parts": [{
                        "function_response": {
                            "name": tool_name,
                            "response": {"content": tool_result}
                        }
                    }]
                })
                # Add tool result to client's display history
                self.conversation_history.append({
                    "role": "tool_output",
                    "parts": [{"text": f"Tool result for {tool_name}: {tool_result}"}]
                })

                # After a tool call and its response, *re-prompt* Gemini with an empty query.
                # This makes Gemini generate the next conversational turn based on the tool result.
                await self.process_query("") # Recursive call for continuation

            elif text_response_content:
                # If no tool call was detected, and there's text content, display it
                self.conversation_history.append({"role": "model", "parts": [{"text": text_response_content}]})
                print(f"Assistant:\n{text_response_content}")
            else:
                logging.warning("Gemini returned no text content or function call for this turn.")
                # This path might be hit if Gemini is still "thinking" or if a previous recursive call handled it.
                if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and not self.conversation_history[-1]["parts"][0].get("text")):
                    self.conversation_history.append({"role": "model", "parts": [{"text": "(No response text)"}]})


        except Exception as e:
            error_message = f"Error processing query with Gemini: {e}"
            logging.error(error_message, exc_info=True) # MODIFIED: Use exc_info=True for proper logging
            # Ensure error is added to history and displayed to user
            if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and self.conversation_history[-1]["parts"][0]["text"].startswith("Error:")):
                self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error_message}"}]})
            print(f"Assistant:\nError: {error_message}")


    async def run_console(self):
        """Starts a console loop for user interaction."""
        logging.info("Research Client ready. Type your queries or 'quit' to exit.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                await self.process_query(user_input)
            except EOFError: # Ctrl-D
                logging.info("EOF detected. Exiting client.")
                break
            except KeyboardInterrupt: # Ctrl-C
                logging.info("Keyboard interrupt detected. Exiting client.")
                break
            except Exception as e:
                logging.error(f"Error in console loop: {e}", exc_info=True) # MODIFIED: Use exc_info=True

async def main():
    logging.info("Starting ResearchClient main function...")
    client = ResearchClient() # API key is now fetched internally by ResearchClient
    try:
        await client.connect()
        await client.run_console()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True) # MODIFIED: Use exc_info=True
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())