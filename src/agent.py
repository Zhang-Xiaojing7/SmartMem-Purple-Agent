import os
import logging
import json

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from purple_agent import PurpleAgent

import json_repair


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("smartmem_purple_agent")


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.purple_agent = PurpleAgent()
        
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        logger.debug(f"{message.context_id=}")
        if not message.context_id: # new test round
            self.purple_agent.reset_memory()
            
        message_text = get_message_text(message)
        agent_response = self.purple_agent.step(message_text)
        
        logger.info("Sending interaction requests...")
            
        # send requests to green
        updater.new_agent_message(
            parts=[Part(root=TextPart(text=agent_response))]
        )
