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
        # Correctly initialize AsyncExitStack once in __init__
        self.exit_stack = AsyncExitStack()
        self._mcp_client: Optional[ClientSession] = None # Type hint for clarity
        self._stop_event = asyncio.Event()

        # Configure Gemini API key
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file or environment.")

        genai.configure(api_key=gemini_api_key)
        # Using gemini-1.5-flash, as it's a generally available and capable model.
        # You can change this to 'gemini-pro' if your API key supports it directly.
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Start chat with empty history initially, will be built up by process_query
        self.chat = self.model.start_chat(history=[])
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
        if not user_query.strip():
            # This path is used for recursive calls after tool execution (continuation)
            pass
        else:
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
                            # 'default' keyword removed as per previous error resolution
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
                        # 'info_fields' removed as per your actual tool implementation
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
            # Send message to Gemini. Gemini's chat object automatically manages history.
            # If `user_query` was empty, it means we are re-prompting Gemini after a tool call.
            # In that case, we send an "empty" message to trigger Gemini to respond
            # based on the newly added tool_response to its history.
            if not user_query.strip() and self.chat.history and self.chat.history[-1]["role"] == "tool":
                response = await self.chat.send_message_async(
                    "", # Send an empty message to trigger Gemini to generate text based on history
                    tools=gemini_tools if gemini_tools else None,
                    stream=True
                )
            elif user_query.strip(): # Only send if there's a new user query
                response = await self.chat.send_message_async(
                    user_query, # Send the actual user query for the new turn
                    tools=gemini_tools if gemini_tools else None,
                    stream=True
                )
            else: # No user query and no pending tool response, nothing to do
                return


            response_content_parts = []
            final_response_text = ""
            function_call_detected = False

            async for chunk in response:
                # Accumulate text content if any
                # MODIFIED: Check parts for text or function_call
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                        response_content_parts.append(part.text)
                        final_response_text += part.text # Accumulate text for display
                    elif hasattr(part, 'function_call') and part.function_call:
                        function_call_detected = True
                        call = part.function_call
                        tool_name = call.name
                        tool_args = dict(call.args) if call.args else {}

                        logging.info(f"Gemini requests tool call: {tool_name} with args: {tool_args}")

                        # Add Gemini's tool call to the chat history (for Gemini's context)
                        self.chat.history.append({
                            "role": "model",
                            "parts": [{
                                "function_call": {"name": tool_name, "args": tool_args}
                            }]
                        })
                        # Add Gemini's tool call to client's display history (optional, for debugging/logging)
                    self.conversation_history.append({
                        "role": "model",
                        "parts": [{"text": f"Calling tool: {tool_name}({tool_args})", "tool_call": tool_args}]
                    })

                    if not self._mcp_client:
                        error_msg = "MCP client not available for tool call. Cannot proceed."
                        logging.error(error_msg)
                        # Add error to Gemini's history so it learns not to call without client
                        self.chat.history.append({"role": "tool", "parts": [{"function_response": {"name": tool_name, "response": {"content": f"Error: {error_msg}"}}}]})
                        self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error_msg}"}]})
                        print(f"\nAssistant:\nError: {error_msg}")
                        return

                    tool_result = await self._mcp_client.call_tool(tool_name, tool_args)
                    logging.info(f"MCP Tool {tool_name} result: {tool_result}")

                    # Add tool response to Gemini's chat history (Crucial for Gemini to process result)
                    self.chat.history.append({
                        "role": "tool",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {
                                    "content": str(tool_result) # Ensure content is a string
                                }
                            }
                        }]
                    })
                    # Add tool result to client's display history
                    self.conversation_history.append({
                        "role": "tool_output",
                        "parts": [{"text": f"Tool result for {tool_name}: {tool_result}"}]
                    })

                    # Recursively call process_query to allow Gemini to process the tool result
                    # Pass empty string to indicate a continuation, not a new user query
                    # REMOVE THESE TWO LINES:
                    # await self.process_query("")
                    # return # Exit this current processing path as a recursive call is handling continuation

            # If no function call was detected and there's accumulated text content from Gemini
            if not function_call_detected and final_response_text:
                # Add Gemini's final text response to conversation history
                self.conversation_history.append({"role": "model", "parts": [{"text": final_response_text}]})
                print(f"Assistant:\n{final_response_text}")
            elif not function_call_detected and not final_response_text:
                logging.warning("Gemini returned no text content or function call for this chunk.")
                # This might happen if a tool was called and the recursive call already handled the response.
                # Or if Gemini truly had no response for a particular turn.
                if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and not self.conversation_history[-1]["parts"][0].get("text")):
                    self.conversation_history.append({"role": "model", "parts": [{"text": "(No response text)"}]})


        except Exception as e:
            error_message = f"Error processing query with Gemini: {e}"
            logging.error(error_message)
            logging.error(traceback.format_exc()) # Log full traceback for debugging
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
                logging.error(f"Error in console loop: {e}")
                logging.error(traceback.format_exc())

async def main():
    logging.info("Starting ResearchClient main function...")
    client = ResearchClient() # API key is now fetched internally by ResearchClient
    try:
        await client.connect()
        await client.run_console()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        logging.error(traceback.format_exc())
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())