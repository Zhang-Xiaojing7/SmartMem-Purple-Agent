import os
import logging
import json

from .memory import get_memory_manager, MemoryItem, ToolInteraction
from .tools import TOOL_SCHEMA
from .prompts import SYSTEM_PROMPT

from openai import OpenAI
import json_repair

system_prompt = SYSTEM_PROMPT
tool_schema = TOOL_SCHEMA

logger = logging.getLogger("smartmem_purple_agent")

class PurpleAgent():
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY") # 直接从环境读取api key和base url, 不再做fallback和分类处理
        base_url = os.getenv('OPENAI_BASE_URL')
        assert api_key and base_url, "Missing API KEY and BASE URL. Please set them in environment."
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        self.model = os.getenv('MODEL_NAME')
        assert self.model, "Please specify the backbone model you want to use."
        
        self.model_generation_args = os.getenv('MODEL_GEN_ARGS', {})
        
        memory_type, memory_args = os.getenv('MEMORY_MANAGER_TYPE', 'fifo'), os.getenv('MEMORY_MANAGER_ARGS', {})
        self.memory = get_memory_manager(memory_type, **memory_args)
        self.tools_schema = tool_schema
        self.memory.set_system_prompt(system_prompt)
        
    def step(self, user_input: str) -> str:
        _parsed = json_repair.loads(user_input)
        isreturnres = isinstance(_parsed, list) and len(_parsed) > 0
        
        # dealing with the memory
        # CASE 1: return tool execution results
        if isreturnres: # it should be a list[dict/str]
            logger.info(f'Received tool results: {_parsed}')
            logger.info("Adding the tool results to memory...")
            
            #TODO: in case loading memory blocks encounters problems
            last_mem_block = self.memory.storage[-1]
            key_to_res_map = {}
            for tr in _parsed:
                metadata = tr.get("metadata", {})
                operation_object = metadata.get("operation_object") # 如果是读取全局状态, 操作对象就是environment
                
                tr.pop("metadata", None)
                res_str = json.dumps(tr, ensure_ascii=False)
                key_to_res_map[operation_object] = res_str # 在没有收到结果的情况下, 如果执行正常, 不会操作一个设备2次以上
                
            for mi in last_mem_block.tool_chain:
                # tool_input: {"device_id": ..., "action": ..., "value": ...}
                device_id = mi['device_id']
                mi.tool_output = key_to_res_map[device_id]
                
            logger.info("Tool results successfully added.")
            
        # CASE 2: green agent sends new instruction
        else:
            logger.info(f'Received Instruction: {user_input}')
            self.memory.add(
                MemoryItem(
                    role="user",
                    content=user_input
                )
            )
            
        running_context = self.memory.get_chat_messages()
        logger.info('Thinking...')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=running_context,
            tools=self.tools_schema if self.tools_schema else None,
            **self.model_generation_args
        )
        msg = response.choices[0].message
        
        # parse the agent response and add it to memory
        if not msg.tool_calls:
            self.memory.add(
                MemoryItem(
                    role="assistant", 
                    content=msg.content,
                    tool_chain=[],
                    token_count=response.usage.completion_tokens
                )
            )
            logger.info(f"[Response]: {msg.content}")
            
            content_text = msg.content if msg.content else ""
            return content_text
        else:
            # add these tool calls to memory and wait for results
            collected_interactions = []
            tool_info_to_send = [] # green only cares about the args
            devices_to_operate = []
            for tool_call in msg.tool_calls: 
                func_name = tool_call.function.name
                call_id = tool_call.id
                raw_args_str = tool_call.function.arguments # {"device_id": ..., "action": ..., "value": ...}
                args = json_repair.loads(raw_args_str)
                devices_to_operate.append(args['device_id'])
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
            
            logger.info(f"Request to operate the devices: {', '.join(devices_to_operate)}")
            
            return json.dumps(tool_info_to_send)
    
    def reset_memory(self):
        """This will clear all the memories except the system prompt."""
        self.memory.clear()