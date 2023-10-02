import json
import logging
from datetime import datetime
from typing import List, Optional

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain.load.serializable import Serializable
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage


logger = logging.getLogger(__name__)

DEFAULT_DBNAME = "chat_history"
DEFAULT_COLLECTION_NAME = "message_store"


class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: Dict[str, Any] = dict()

                @property
                def memory_variables(self) -> List[str]:
                    return list(self.memories.keys())

                def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    return self.memories

                def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                    pass

                def clear(self) -> None:
                    pass
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this chain run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A list of Messages stored in-memory."""

    def add_user_message(
        self, message: str, standalone_question: Optional[str] = None
    ) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message), standalone_question)

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        self.add_message(AIMessage(content=message))

    @abstractmethod
    def add_message(self, message: BaseMessage, standalone_question: str) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in MongoDB.

    Args:
        connection_string: connection string to connect to MongoDB
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        database_name: name of the database to use
        collection_name: name of the collection to use
    """

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str = DEFAULT_DBNAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        from pymongo import MongoClient, errors

        self.connection_string = connection_string
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            self.client: MongoClient = MongoClient(connection_string)
        except errors.ConnectionFailure as error:
            logger.error(error)

        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.collection.create_index("SessionId")

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from MongoDB"""
        from pymongo import errors

        try:
            cursor = self.collection.find({"SessionId": self.session_id})
        except errors.OperationFailure as error:
            logger.error(error)

        if cursor:
            items = [json.loads(document["History"]) for document in cursor]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_message(
        self, message: BaseMessage, standalone_question: Optional[str] = None
    ) -> None:
        """Append the message to the record in MongoDB"""
        from pymongo import errors

        try:
            self.collection.insert_one(
                {
                    "SessionId": self.session_id,
                    "History": json.dumps(_message_to_dict(message)),
                    "StandaloneQuestion": standalone_question,
                    "Timestamp": datetime.now(),
                }
            )
        except errors.WriteError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from MongoDB"""
        from pymongo import errors

        try:
            self.collection.delete_many({"SessionId": self.session_id})
        except errors.WriteError as err:
            logger.error(err)
