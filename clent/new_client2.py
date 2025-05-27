import asyncio
import logging
import os
import functools
import sys # <--- ADD THIS IMPORT
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
import httpx
import anyio
from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
import google.generativeai as genai
from mcp import ClientSession
# from mcp.client.sse import sse_client # <--- COMMENT OUT OR REMOVE THIS IMPORT
from dotenv import load_dotenv

load_dotenv()

# Define prompt_toolkit styles
PROMPT_STYLE_DICT = {
    "prompt": "fg:yellow",
    "output.model": "fg:green",
    "output.tool": "fg:blue",
    "output.error": "fg:red",
    "output.warning": "fg:yellow",
    "output.debug": "fg:gray",
}
PROMPT_STYLE_OBJ = Style.from_dict(PROMPT_STYLE_DICT)

# Custom logging handler for prompt_toolkit
class PromptToolkitLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            log_entry = self.format(record)
            style_class = "output.debug"
            if record.levelno >= logging.ERROR:
                style_class = "output.error"
            elif record.levelno >= logging.WARNING:
                style_class = "output.warning"
            elif record.levelno >= logging.INFO:
                style_class = "output.debug"
            if log_entry.strip():
                print_pt(log_entry.strip(), style_class=style_class)
        except Exception:
            self.handleError(record)

def setup_logging(debug: bool = False):
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    pt_handler = PromptToolkitLogHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    pt_handler.setFormatter(formatter)
    root_logger.addHandler(pt_handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

def print_pt(text: str, style_class: str = ""):
    if style_class:
        print_formatted_text(FormattedText([(f"class:{style_class}", text)]), style=PROMPT_STYLE_OBJ)
    else:
        print_formatted_text(text)

def truncate_text(text: str, max_length: int = 250):
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length//2] + "..." + text[-max_length//2:]

# Decorator for retryable async functions
def retryable(max_retries=3, delay=1, connection_errors=(httpx.ReadError, httpx.ConnectError,
                                                        anyio.ClosedResourceError, ConnectionError)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            operation_name = func.__name__
            retries = 0
            while True:
                try:
                    return await func(self, *args, **kwargs)
                except connection_errors as e:
                    retries += 1
                    if retries >= max_retries:
                        error_message = f"{operation_name} failed after {retries} attempts: {e}"
                        print_pt(error_message, "output.error")
                        return error_message
                    print_pt(f"Connection error: {e}. Attempting reconnect... ({retries}/{max_retries})", "output.error")
                    await asyncio.sleep(delay)
                except Exception as error:
                    error_message = f"Error processing {operation_name}: {error}"
                    print_pt(error_message, "output.error")
                    return error_message
        return wrapper
    return decorator

class MCPGeminiClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.exit_stack = AsyncExitStack()
        self._stop_event = asyncio.Event()
        self._mcp_client: Optional[ClientSession] = None
        self.tools = []
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.conversation_history = []
        self.prompt_session = PromptSession(history=None)
        self.pending_tool_call = None

    @retryable(max_retries=5, delay=1)
    async def connect(self):
        """Establish stdio connection to MCP server with retries"""
        try:
            print_pt(f"[DEBUG] Attempting to clean up any previous connections...", "output.debug")
            await self.cleanup()
            print_pt(f"[DEBUG] Attempting to connect to MCP server via stdio", "output.debug")

            # Create async file streams for stdin and stdout
            # These streams provide async read/write methods compatible with ClientSession
            read_stream = anyio.streams.file.FileReadStream(sys.stdin.buffer)
            write_stream = anyio.streams.file.FileWriteStream(sys.stdout.buffer)
            
            # Enter the streams into the exit stack to ensure they are closed cleanly
            await self.exit_stack.enter_async_context(read_stream)
            await self.exit_stack.enter_async_context(write_stream)

            print_pt(f"[DEBUG] Stdio connection established. Creating ClientSession...", "output.debug")
            # ClientSession expects async read/write functions
            self._mcp_client = await self.exit_stack.enter_async_context(ClientSession(read_stream.read, write_stream.write))
            
            print_pt(f"[DEBUG] Initializing MCP client...", "output.debug")
            await self._mcp_client.initialize()
            print_pt(f"[DEBUG] Listing tools from MCP server...", "output.debug")
            self.tools = await self._mcp_client.list_tools()  # Returns a list
            print_pt(f"Connected to MCP server. Available tools: {[tool.name for tool in self.tools]}", "output.debug")
            return True
        except Exception as e:
            print_pt(f"[ERROR] Connection error: {e}", "output.error")
            import traceback
            print_pt(traceback.format_exc(), "output.error")
            raise

    async def cleanup(self):
        """Clean up resources and connections"""
        print_pt("Cleaning up resources...", "output.debug")
        self._stop_event.set()
        await self.exit_stack.aclose()
        self._mcp_client = None
        print_pt("Cleanup complete.", "output.debug")

    async def process_query(self, user_query: str):
        if not user_query.strip() and not self.conversation_history:
            print_pt("No query provided.", "output.warning")
            return

        if user_query.strip():
            if self.pending_tool_call and isinstance(self.pending_tool_call, dict):
                try:
                    provided_values = [v.strip() for v in user_query.split(',')]
                    tool_schema = next((t for t in self.tools if t.name == self.pending_tool_call["name"]), None)
                    if tool_schema and 'properties' in tool_schema.parameters:
                        required_params = tool_schema.parameters.get('required', [])
                        missing_params = [p for p in required_params if p not in self.pending_tool_call["args"]]
                        for i, param_name in enumerate(missing_params):
                            if i < len(provided_values):
                                self.pending_tool_call["args"][param_name] = provided_values[i]
                            else:
                                break
                        still_missing = [p for p in required_params if p not in self.pending_tool_call["args"]]
                        if not still_missing:
                            print_pt(f"Resuming tool call for {self.pending_tool_call['name']} with updated args: {self.pending_tool_call['args']}", "output.debug")
                            self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
                            user_query = ""
                        else:
                            clarification = f"Still need: {', '.join(still_missing)} for {self.pending_tool_call['name']}. You provided: {user_query}"
                            self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
                            self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                            return clarification
                    else:
                        print_pt(f"Error: Tool schema not found for pending call {self.pending_tool_call['name']}", "output.error")
                        self.pending_tool_call = None
                except Exception as e:
                    print_pt(f"Error processing input for pending tool call: {e}", "output.error")
                    clarification = f"Error processing your input for {self.pending_tool_call['name']}. Please provide: {', '.join(missing_params)}"
                    self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
                    self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                    return clarification
            else:
                self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})

        gemini_tools = []
        for tool in self.tools:
            params_schema = getattr(tool, 'parameters', {})
            if isinstance(params_schema, dict) and 'type' in params_schema and 'properties' in params_schema:
                gemini_tools.append({
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params_schema
                    }]
                })
            else:
                print_pt(f"Warning: Tool '{tool.name}' parameters might not be in expected schema or are missing. Actual schema: {params_schema}", "output.warning")
                gemini_tools.append({
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description
                    }]
                })

        try:
            if self.pending_tool_call and isinstance(self.pending_tool_call, dict) and self.pending_tool_call.get("name") and self.pending_tool_call.get("args") is not None:
                tool_name_to_call = self.pending_tool_call["name"]
                tool_args_to_call = self.pending_tool_call["args"]
                tool_schema = next((t for t in self.tools if t.name == tool_name_to_call), None)
                if tool_schema and 'properties' in getattr(tool_schema, 'parameters', {}):
                    required_params = tool_schema.parameters.get('required', [])
                    missing_params = [p for p in required_params if p not in tool_args_to_call]
                    if missing_params:
                        clarification = f"Please provide the following for {tool_name_to_call}: {', '.join(missing_params)}"
                        if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and self.conversation_history[-1]["parts"][0]["text"].startswith("Please provide")):
                            self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                        return clarification
                print_pt(f"Executing pending tool call: {tool_name_to_call} with args: {tool_args_to_call}", "output.tool")
                if not self._mcp_client:
                    error = "MCP client not available for tool call."
                    print_pt(error, "output.error")
                    self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error}"}]})
                    self.pending_tool_call = None
                    return error
                tool_result = await self._mcp_client.call_tool(tool_name_to_call, tool_args_to_call)
                print_pt(f"MCP Tool {tool_name_to_call} result: {truncate_text(str(tool_result))}", "output.tool")
                self.conversation_history.append({
                    "role": "user",
                    "parts": [{"function_call": {"name": tool_name_to_call, "args": tool_args_to_call}}]
                })
                self.conversation_history.append({
                    "role": "model",
                    "parts": [{"function_response": {"name": tool_name_to_call, "response": {"content": str(tool_result)}}}]
                })
                self.pending_tool_call = None
                return await self.process_query("")

            current_chat_session = self.model.start_chat(history=self.conversation_history)
            content_to_send = self.conversation_history[-1]
            response = await current_chat_session.send_message_async(
                content_to_send,
                tools=gemini_tools if gemini_tools else None,
            )
            response_content = ""
            function_call_detected = False
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        response_content += part.text
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call_detected = True
                        call = part.function_call
                        tool_name = call.name
                        tool_args = dict(call.args) if call.args else {}
                        tool_schema = next((t for t in self.tools if t.name == tool_name), None)
                        if tool_schema and isinstance(getattr(tool_schema, 'parameters', None), dict) and 'properties' in tool_schema.parameters:
                            required_params = tool_schema.parameters.get('required', [])
                            missing_params = [p for p in required_params if p not in tool_args]
                            if missing_params:
                                self.pending_tool_call = {"name": tool_name, "args": tool_args}
                                clarification = f"Please provide the following for {tool_name}: {', '.join(missing_params)}"
                                self.conversation_history.append({
                                    "role": "model",
                                    "parts": [{
                                        "function_call": {"name": tool_name, "args": tool_args}
                                    }]
                                })
                                self.conversation_history.append({
                                    "role": "model",
                                    "parts": [{"text": clarification}]
                                })
                                return clarification
                        print_pt(f"Gemini requests tool call: {tool_name} with args: {tool_args}", "output.tool")
                        if not self._mcp_client:
                            error = "MCP client not available for tool call."
                            print_pt(error, "output.error")
                            self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error}"}]})
                            return error
                        tool_result = await self._mcp_client.call_tool(tool_name, tool_args)
                        print_pt(f"MCP Tool {tool_name} result: {truncate_text(str(tool_result))}", "output.tool")
                        self.conversation_history.append({
                            "role": "model",
                            "parts": [{
                                "function_call": {"name": tool_name, "args": tool_args}
                            }]
                        })
                        self.conversation_history.append({
                            "role": "model",
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {
                                        "content": str(tool_result)
                                    }
                                }
                            }]
                        })
                        self.pending_tool_call = None
                        return await self.process_query("")
            if not function_call_detected:
                final_content = response_content.strip()
                if final_content:
                    self.conversation_history.append({"role": "model", "parts": [{"text": final_content}]})
                    self.pending_tool_call = None
                    return final_content
                else:
                    if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and not self.conversation_history[-1]["parts"][0].get("text")):
                        print_pt("Gemini returned no text content or function call.", "output.warning")
                        self.conversation_history.append({"role": "model", "parts": [{"text": "(No response text)"}]})
                    return "(No response text)"
        except Exception as e:
            error_message = f"Error processing query with Gemini: {e}"
            print_pt(error_message, "output.error")
            if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and self.conversation_history[-1]["parts"][0]["text"].startswith("Error:")):
                self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error_message}"}]})
            import traceback
            print_pt(traceback.format_exc(), "output.error")
            self.pending_tool_call = None
            return error_message

    async def chat_loop(self):
        print_pt("MCP Gemini Client started!", "output.model")
        print_pt("Type your queries or 'quit' to exit.", "output.model")
        while True:
            try:
                prompt_message = "User: "
                if self.pending_tool_call and isinstance(self.pending_tool_call, dict):
                    tool_name = self.pending_tool_call['name']
                    tool_schema = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_schema and 'properties' in getattr(tool_schema, 'parameters', {}):
                        required_params = tool_schema.parameters.get('required', [])
                        missing_params = [p for p in required_params if p not in self.pending_tool_call['args']]

                        if missing_params:
                            prompt_message = f"Input for {tool_name} ({', '.join(missing_params)}): "
                query = await self.prompt_session.prompt_async(
                    FormattedText([("class:prompt", prompt_message)]),
                    style=PROMPT_STYLE_OBJ
                )
                query = query.strip()
                if query.lower() in ('quit', 'exit'):
                    break
                response = await self.process_query(query)
                if response:
                    print_pt(f"Assistant:\n{response}", "output.model")
            except (EOFError, KeyboardInterrupt):
                print_pt("Exiting client...", "output.debug")
                break
            except Exception as e:
                print_pt(f"Error in chat loop: {e}", "output.error")
                import traceback
                print_pt(traceback.format_exc(), "output.error")

async def main():
    setup_logging(debug=True)
    server_url = "stdio"  # This value is now just a placeholder/indicator, not used for actual connection logic
    if not os.getenv("GEMINI_API_KEY"):
        print_pt("Error: GEMINI_API_KEY not found in .env file.", "output.error")
        return
    client = MCPGeminiClient(server_url=server_url)
    try:
        if not await client.connect():
            print_pt("Initial connection to MCP server failed. Exiting.", "output.error")
            return
        with patch_stdout():
            await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print_pt(f"\nUnhandled exception occurred: {e}", "output.error")
        import traceback
        print_pt(traceback.format_exc(), "output.error")