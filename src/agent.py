import os
import logging
import json

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from memory import get_memory_manager, MemoryItem, ToolInteraction
#TODO: for tool schema and system prompt, maybe we can add more versions... and we can use factory to flexiably get the specific one...
from tools import TOOL_SCHEMA
from prompts import SYSTEM_PROMPT

from openai import OpenAI
# from json_repair import repair_json
import json_repair

system_prompt = SYSTEM_PROMPT
tool_schema = TOOL_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartmem_purple_agent")

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        
        # Support multiple API providers
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if google_api_key:
            # Use Gemini via OpenAI-compatible endpoint
            api_key = google_api_key
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            default_model = "gemini-2.0-flash"
            logger.info("Using Google Gemini API")
        elif openai_api_key:
            api_key = openai_api_key
            base_url = os.getenv('OPENAI_BASE_URL', "https://api.openai.com/v1")
            default_model = "gpt-4o"
            logger.info("Using OpenAI API")
        else:
            raise ValueError("No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in environment.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv('MODEL_NAME', default_model)
        self.model_generation_args = os.getenv('MODEL_GEN_ARGS', {})
        
        memory_type, memory_args = os.getenv('MEMORY_MANAGER_TYPE', 'fifo'), os.getenv('MEMORY_MANAGER_ARGS', {})
        self.memory = get_memory_manager(memory_type, **memory_args)
        self.tools_schema = tool_schema
        system_mem = MemoryItem(
                role="system",
                raw_data={"role": "system", "content": system_prompt},
                content=system_prompt
            )
        self.memory.add(system_mem)
        self.max_tool_use_iter = os.getenv('MAX_TOOL_USE_ITER', 3) #TODO: make it useful later...
        
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        message_text = get_message_text(message)
        
        _parsed = json_repair.loads(message_text)
        isreturnres = isinstance(_parsed, list) and len(_parsed) > 0
        
        # dealing with the memory
        # CASE 1: return tool execution results
        if isreturnres: # it should be a list[dict/str]
            logger.info("Adding the tool results to memory...")
            try:
                tool_res = json.load(message.parts.root.data)['tool_results'] # datafile should be dict[str, any], so we should use {"tool_results": [...]} ?
            except Exception as e:
                logger.error("Failed to load the tool results.")
            #TODO: in case loading memory blocks encounters problems
            last_mem_block = self.memory.storage[-1]
            for i, mi in last_mem_block.tool_chain: #TODO: make sure the returned results match the sent tool calls
                mi.tool_output = tool_res[i]
            logger.info("Tool results successfully added.")
            
        # CASE 2: green agent sends new instruction
        else:
            self.memory.add(
                MemoryItem(
                    role=message.role,
                    content=message_text
                )
            )

        running_context = self.memory.get_chat_messages() # a temp context during multi-tool calls for sloving one user query
        
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=running_context,
            tools=self.tools_schema if self.tools_schema else None,
            **self.model_generation_args
        )
        msg = response.choices[0].message
        running_context.append(msg)
        
        if not msg.tool_calls:
            self.memory.add(
                MemoryItem(
                    role="assistant", 
                    content=msg.content,
                    tool_chain=[],  # Empty list instead of None
                    token_count=response.usage.completion_tokens
                )
            )
            logger.info("[Response]: {msg.content}")
            # Ensure content is not None before sending
            content_text = msg.content if msg.content else ""
            updater.new_agent_message(parts=[Part(root=TextPart(text=content_text))])
            # await updater.add_artifact(
            #     parts=[Part(root=TextPart(text=msg.content))],
            #     name="Response",
            #     metadata={"message_type": "text_message"}
            # )
            return
        else:
            # add these tool calls to memory and wait for results
            collected_interactions = []
            tool_info_to_send = [] # green only cares about the args
            for tool_call in msg.tool_calls: 
                func_name = tool_call.function.name
                call_id = tool_call.id
                raw_args_str = tool_call.function.arguments
                args = json_repair.loads(raw_args_str)
                tool_info_to_send.append(args)

                ti = ToolInteraction(
                    tool_name=func_name,
                    tool_id=call_id,
                    tool_input=args, 
                    tool_output=""
                )
                collected_interactions.append(ti)
                
            self.memory.add(MemoryItem(
                role="assistant", 
                content=msg.content,
                tool_chain=collected_interactions
            ))
            logger.info("Sending interaction requests...")
            
            # send requests to green
            # Ensure content is not None before sending
            content_text = msg.content if msg.content else ""
            tool_msg = {"tool_calls": tool_info_to_send}
            updater.new_agent_message(
                parts=[Part(root=TextPart(text=content_text)), Part(root=DataPart(data=tool_msg))]
            )
