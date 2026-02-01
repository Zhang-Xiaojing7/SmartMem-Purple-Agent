import logging

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from purple_agent.agent import PurpleAgent


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
        agent_response = await self.purple_agent.step(message_text)

        logger.info(f"Sending response: {agent_response}")

        # Complete the task with the response
        await updater.complete(new_agent_text_message(agent_response))
