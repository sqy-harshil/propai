import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Constants
FAISS_VECTORSTORE_PATH = "generic_content_index/"
DATABASE_NAME = "propai"
LISTINGS_HISTORY = "chat_history"
SUPPLEMENTARY_HISTORY = "supplementary"
PROPERTIES = "properties"
LISTINGS_WINDOW_MEMORY = 5
SUPPLEMENTARY_WINDOW_MEMORY = 5
NUM_MESSAGES_TO_RETRIEVE = 4

# Azure Configurations
AZURE_COMMON_PARAMS = {
    "openai_api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "openai_api_type": os.getenv("AZURE_OPENAI_API_TYPE"),
}

AZURE_GPT35_PARAMS = {
    "streaming": True,
    "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT_35"),
    **AZURE_COMMON_PARAMS,
}

AZURE_GPT4_PARAMS = {
    "streaming": True,
    "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT_4"),
    **AZURE_COMMON_PARAMS,
}

AZURE_ADA_PARAMS = {
    "deployment": os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
    **AZURE_COMMON_PARAMS,
}

CLASSIFIER_PARAMS = {
    "engine": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT_4"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
    "api_type": os.getenv("AZURE_OPENAI_API_TYPE"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
}

# Pinecone Configurations
PINECONE_PARAMS = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "environment": os.getenv("PINECONE_ENV"),
}
INDEX_NAME = os.getenv("PINECONE_INDEX")

# MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")
