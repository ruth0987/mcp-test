import asyncio
import logging
import os
import functools
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
import httpx
import anyio
from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

import google.generativeai as genai # Added Gemini import
from modelcontextprotocol.client.aio import Client, StdioTransport

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
            style_class = "output.debug"  # Default

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
                        return error_message # Or raise, depending on desired behavior
                    
                    print_pt(f"Connection error: {e}. Attempting reconnect... ({retries}/{max_retries})", "output.error")
                    await asyncio.sleep(delay)
                except Exception as error:
                    error_message = f"Error processing {operation_name}: {error}"
                    print_pt(error_message, "output.error")
                    return error_message # Or raise
                    
        return wrapper
    return decorator

class MCPGeminiClient: # Renamed from MCPGroqClient
    def __init__(self, server_command: str):
        self.server_command = server_command
        self.exit_stack = AsyncExitStack()
        self._stop_event = asyncio.Event()
        
        self._mcp_client: Optional[Client] = None # Type hint for clarity
        self.tools = []
        
        # Gemini initialization
        genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.conversation_history = [] # Moved from ConversationManager
        
        self.prompt_session = PromptSession(history=None)
        self.pending_tool_call = None  # Store incomplete tool call context 
        
    @retryable(max_retries=5, delay=1)
    async def connect(self):
        """Establish connection to MCP server with retries"""
        try:
            await self.cleanup()  # Clean up any existing connections
            
            transport = StdioTransport(cmd=self.server_command.split())
            self._mcp_client = await self.exit_stack.enter_async_context(Client(transport))
            
            await self._mcp_client.initialize_session()
            self.tools = await self._mcp_client.get_tools()
            
            print_pt(f"Connected to MCP server. Available tools: {[tool.name for tool in self.tools]}", "output.debug")
            return True
        except Exception as e:
            print_pt(f"Connection error: {e}", "output.error")
            raise
    
    async def cleanup(self):
        """Clean up resources and connections"""
        print_pt("Cleaning up resources...", "output.debug")
        self._stop_event.set()
        await self.exit_stack.aclose()
        self._mcp_client = None
        print_pt("Cleanup complete.", "output.debug")
    
    # process_query as provided by user, integrated into this class
    async def process_query(self, user_query: str): 
        if not user_query.strip() and not self.conversation_history: 
            print_pt("No query provided.", "output.warning") 
            return 

        if user_query.strip(): 
            # If there's a pending tool call, try to complete its arguments
            if self.pending_tool_call and isinstance(self.pending_tool_call, dict):
                # Simple heuristic: assume user query is providing missing info
                # This might need more sophisticated parsing in a real scenario
                # For now, let's assume the user query is a comma-separated list of values
                # for the missing parameters.
                try:
                    provided_values = [v.strip() for v in user_query.split(',')]
                    tool_schema = next((t for t in self.tools if t.name == self.pending_tool_call["name"]), None)
                    if tool_schema and 'properties' in tool_schema.parameters:
                        required_params = tool_schema.parameters.get('required', [])
                        missing_params = [p for p in required_params if p not in self.pending_tool_call["args"] or not self.pending_tool_call["args"][p]]
                        
                        for i, param_name in enumerate(missing_params):
                            if i < len(provided_values):
                                self.pending_tool_call["args"][param_name] = provided_values[i]
                            else:
                                break # Not enough values provided
                        
                        # Check if all missing params are now filled
                        still_missing = [p for p in required_params if p not in self.pending_tool_call["args"] or not self.pending_tool_call["args"][p]]
                        if not still_missing:
                            # All params gathered, proceed with the stored tool call context
                            print_pt(f"Resuming tool call for {self.pending_tool_call['name']} with updated args: {self.pending_tool_call['args']}", "output.debug")
                            # Effectively, we are now ready to make the tool call part of the original function call logic
                            # The original logic will pick this up. We just need to ensure conversation history is correct.
                            # The user's input that provided missing params should be part of the history.
                            self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
                            user_query = "" # Clear user_query as we are now processing the stored context
                        else:
                            # Still missing parameters, re-prompt
                            clarification = f"Still need: {', '.join(still_missing)} for {self.pending_tool_call['name']}. You provided: {user_query}"
                            self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]}) # Add user's attempt
                            self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                            return clarification
                    else:
                        # Tool schema not found, should not happen if pending_tool_call is set correctly
                        print_pt(f"Error: Tool schema not found for pending call {self.pending_tool_call['name']}", "output.error")
                        self.pending_tool_call = None # Clear invalid pending call

                except Exception as e:
                    print_pt(f"Error processing input for pending tool call: {e}", "output.error")
                    # Keep pending_tool_call as is, or ask again
                    clarification = f"Error processing your input for {self.pending_tool_call['name']}. Please provide: {', '.join(missing_params)}"
                    self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
                    self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                    return clarification
            else:
                 self.conversation_history.append({"role": "user", "parts": [{"text": user_query}]})
        elif self.pending_tool_call and isinstance(self.pending_tool_call, dict) and not user_query.strip():
            # This case means we are re-running process_query, likely after a successful tool call,
            # but there was a pending tool call that needed more info. This shouldn't ideally happen
            # if the logic is correct, as a successful tool call should clear pending_tool_call.
            # Or, it's the first run after a tool call that *was* pending but now has args.
            pass # Let the logic proceed to make the call if args are complete.

        # Convert MCP tools to Gemini function format 
        gemini_tools = [] 
        for tool in self.tools: 
            params_schema = getattr(tool, 'parameters', {}) 
            # Ensure params_schema is a dictionary before accessing keys
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
                        # Gemini might require a parameters object even if empty, e.g., "parameters": {"type": "object", "properties": {}} 
                    }] 
                }) 

        try: 
            # Determine what to send to Gemini
            # If there's a pending tool call with all arguments now presumably filled,
            # we should proceed as if Gemini just returned that tool call request.
            # Otherwise, send the latest user message or continue the conversation.
            if self.pending_tool_call and isinstance(self.pending_tool_call, dict) and self.pending_tool_call.get("name") and self.pending_tool_call.get("args") is not None:
                tool_name_to_call = self.pending_tool_call["name"]
                tool_args_to_call = self.pending_tool_call["args"]
                tool_schema = next((t for t in self.tools if t.name == tool_name_to_call), None)
                
                if tool_schema and 'properties' in getattr(tool_schema, 'parameters', {}):
                    required_params = tool_schema.parameters.get('required', [])
                    missing_params = [p for p in required_params if p not in tool_args_to_call or not tool_args_to_call[p]]
                    if missing_params:
                        # This means we tried to fill args but failed, and the user didn't provide more.
                        # This case should have been handled by the input processing block above.
                        # If we reach here, it's likely a logic gap or an empty re-query after a failed fill.
                        clarification = f"Please provide the following for {tool_name_to_call}: {', '.join(missing_params)}"
                        # Avoid adding duplicate model requests for clarification if history already has it.
                        if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and self.conversation_history[-1]["parts"][0]["text"].startswith("Please provide")):
                            self.conversation_history.append({"role": "model", "parts": [{"text": clarification}]})
                        return clarification
                # If all good, proceed to the tool call execution block as if Gemini just requested it.
                # This bypasses sending a message to Gemini again for this turn.
                print_pt(f"Executing pending tool call: {tool_name_to_call} with args: {tool_args_to_call}", "output.tool")
                if not self._mcp_client:
                    error = "MCP client not available for tool call."
                    print_pt(error, "output.error")
                    self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error}"}]})
                    self.pending_tool_call = None # Clear to avoid loop
                    return error

                tool_result = await self._mcp_client.call_tool(tool_name_to_call, tool_args_to_call)
                print_pt(f"MCP Tool {tool_name_to_call} result: {truncate_text(str(tool_result))}", "output.tool")

                self.conversation_history.append({
                    "role": "user", # Gemini expects user role for the function call part that initiated the response
                    "parts": [{"function_call": {"name": tool_name_to_call, "args": tool_args_to_call}}]
                })
                self.conversation_history.append({
                    "role": "model", # Gemini expects model role for the function response part
                    "parts": [{"function_response": {"name": tool_name_to_call, "response": {"content": str(tool_result)}}}]
                })
                self.pending_tool_call = None # Clear pending call
                return await self.process_query("") # Re-run with tool results, sending history to Gemini

            # If no pending tool call to execute directly, send message to Gemini
            current_chat_session = self.model.start_chat(history=self.conversation_history)
            # Determine content to send: if user_query is non-empty, use it.
            # Otherwise, if history exists, it implies a follow-up (e.g., after tool use), so send the last content.
            # The self.conversation_history should be up-to-date here.
            content_to_send = self.conversation_history[-1] # Send the last part of the history
            
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
                            missing_params = [p for p in required_params if p not in tool_args or not tool_args[p]] # Check for empty strings too
                            if missing_params:
                                self.pending_tool_call = {"name": tool_name, "args": tool_args} # Store current args
                                clarification = f"Please provide the following for {tool_name}: {', '.join(missing_params)}"
                                # Append Gemini's request for tool call to history
                                self.conversation_history.append({
                                    "role": "model", 
                                    "parts": [{
                                        "function_call": {"name": tool_name, "args": tool_args}
                                    }]
                                })
                                # Append our clarification message to history
                                self.conversation_history.append({
                                    "role": "model", # Or assistant, representing the system asking for more info
                                    "parts": [{"text": clarification}]
                                })
                                return clarification

                        # All required parameters are present, execute tool call
                        print_pt(f"Gemini requests tool call: {tool_name} with args: {tool_args}", "output.tool")
                        if not self._mcp_client:
                            error = "MCP client not available for tool call."
                            print_pt(error, "output.error")
                            self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error}"}]})
                            return error

                        tool_result = await self._mcp_client.call_tool(tool_name, tool_args)
                        print_pt(f"MCP Tool {tool_name} result: {truncate_text(str(tool_result))}", "output.tool")
                        
                        # Append Gemini's actual function call part that led to this
                        self.conversation_history.append({
                            "role": "model", # This was Gemini's turn that resulted in a tool call
                            "parts": [{
                                "function_call": {"name": tool_name, "args": tool_args}
                            }]
                        })
                        # Append the tool's response, making sure it's in the format Gemini expects for a function response
                        self.conversation_history.append({
                            "role": "model", # For Gemini, this is a 'function' role or 'tool' role part
                                          # Using 'model' and structuring 'parts' for function_response
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {
                                        "content": str(tool_result) 
                                    }
                                }
                            }]
                        })
                        self.pending_tool_call = None # Clear pending call as it's now handled
                        return await self.process_query("") # Re-run with tool results

            if not function_call_detected:
                final_content = response_content.strip()
                if final_content:
                    self.conversation_history.append({"role": "model", "parts": [{"text": final_content}]})
                    self.pending_tool_call = None # Clear any pending call
                    return final_content
                else:
                    # This case might occur if Gemini responds with an empty message after tool execution or for other reasons.
                    # Avoid adding an empty model message if the last message was already an empty model response.
                    if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and not self.conversation_history[-1]["parts"][0].get("text")):
                        print_pt("Gemini returned no text content or function call.", "output.warning")
                        self.conversation_history.append({"role": "model", "parts": [{"text": "(No response text)"}]})
                    return "(No response text)"

        except Exception as e:
            error_message = f"Error processing query with Gemini: {e}"
            print_pt(error_message, "output.error")
            # Add error to history to inform the model if it continues
            if not (self.conversation_history and self.conversation_history[-1]["role"] == "model" and self.conversation_history[-1]["parts"][0]["text"].startswith("Error:")):
                self.conversation_history.append({"role": "model", "parts": [{"text": f"Error: {error_message}"}]})
            import traceback
            print_pt(traceback.format_exc(), "output.error")
            self.pending_tool_call = None # Clear pending call on error
            return error_message

    async def chat_loop(self):
        """Run interactive chat loop"""
        print_pt("MCP Gemini Client started!", "output.model")
        print_pt("Type your queries or 'quit' to exit.", "output.model")
        
        while True:
            try:
                # Display pending tool prompt if any
                prompt_message = "User: "
                if self.pending_tool_call and isinstance(self.pending_tool_call, dict):
                    tool_name = self.pending_tool_call['name']
                    tool_schema = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_schema and 'properties' in getattr(tool_schema, 'parameters', {}):
                        required_params = tool_schema.parameters.get('required', [])
                        missing_params = [p for p in required_params if p not in self.pending_tool_call['args'] or not self.pending_tool_call['args'][p]]
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
                if response: # response can be None if process_query returns early
                    print_pt(f"Assistant:\n{response}", "output.model")
                    
            except (EOFError, KeyboardInterrupt):
                print_pt("Exiting client...", "output.debug")
                break
            except Exception as e:
                print_pt(f"Error in chat loop: {e}", "output.error")
                import traceback
                print_pt(traceback.format_exc(), "output.error")
                # Optionally, decide if to break or continue loop

async def main():
    setup_logging(debug=True)
    
    # Construct path to server script (new_tools.py)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_script_path = os.path.join(project_root, "server", "tools", "new_tools.py")
    server_command = f"python {server_script_path}"
    
    if not os.getenv("GEMINI_API_KEY"):
        print_pt("Error: GEMINI_API_KEY not found in .env file.", "output.error")
        return

    client = MCPGeminiClient(server_command=server_command)
    
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