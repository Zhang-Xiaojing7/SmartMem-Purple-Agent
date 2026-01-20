from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import random
import json


def generate_readable_id() -> str:
    """
    Generates a human-readable ID strictly for debug and research analysis.
    Format: YYYYMMDD_HHMMSS_RANDOM (e.g., 20231027_103005_1234)
    This allows files/logs to be naturally sorted by time.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = random.randint(1000, 9999)
    return f"{timestamp}_{suffix}"

class ToolInteraction(BaseModel):
    tool_name: str
    tool_id: str
    tool_input: Union[str, Dict[str, Any]]
    tool_output: str
    
    def to_string(self):
        return f"[Call: {self.tool_name}({self.tool_input})] -> [Result: {self.tool_output}]"

class MemoryItem(BaseModel):
    id: str = Field(default_factory=generate_readable_id)
    timestamp: datetime = Field(default_factory=datetime.now)
    role: str  # 'user', 'assistant' or 'system'
    
    content: Optional[str] = None

    tool_chain: List[ToolInteraction] = Field(default_factory=list)

    raw_data: Any = None
    
    token_count: Optional[int] = Field(
        default=None, 
        description="Token count from LLM API response or estimated by Memory."
    )

    def get_display_content(self) -> str:
        """
        get text memory.
        """
        parts = []
        if self.tool_chain:
            parts.extend([t.to_string() for t in self.tool_chain])
        if self.content:
            parts.append(self.content)
            
        return "\n".join(parts)
    
    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """
        Restore MemoryItem back to the original message list required by OpenAI.
        If it is a normal conversation, return [1 message].
        If it is a tool call chain, return [Assistant message, Tool message 1, Tool message 2...]
        """
        
        # CASE 1: this is a normal conversation
        if not self.tool_chain:
            return [{
                "role": self.role,
                "content": self.content
            }]

        # CASE 2: This is a tool call round (Assistant calls a tool -> Tool returns)
        # According to OpenAI's specifications, we need to construct:
        # 1. An Assistant message containing all tool calls
        # 2. Followed by several corresponding Tool messages
        
        messages = []
        
        openai_tool_calls = []
        for interaction in self.tool_chain:
            openai_tool_calls.append({
                "id": interaction.tool_id,
                "type": "function",
                "function": {
                    "name": interaction.tool_name,
                    "arguments": json.dumps(interaction.tool_input) if isinstance(interaction.tool_input, dict) else interaction.tool_input
                }
            })
            
        assistant_msg = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": openai_tool_calls
        }
        messages.append(assistant_msg)
        
        for interaction in self.tool_chain:
            tool_msg = {
                "role": "tool",
                "tool_call_id": interaction.tool_id,
                "content": interaction.tool_output
            }
            messages.append(tool_msg)
            
        return messages