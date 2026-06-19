from dataclasses import dataclass
import os

from dotenv import load_dotenv
from typing import Union

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

import openai
import json
import warnings
from pprintpp import pprint as pp

def enable_logging():
    import logging

    logging.disable(logging.NOTSET)

    # root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    # for h in root.handlers:
    #     h.setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG, force=True) # to show the communication embedding model and the vector store

def disable_logging():
    import logging

    logging.disable(logging.CRITICAL)
    
# This dataclass holds the Databricks configuration loaded from environment variables.
# The `frozen=True` parameter makes it immutable, which is a good practice for configuration objects.
@dataclass(frozen=True)
class DatabricksConfig:
    token: str
    host: str
    endpoint: str


class MissingEnvironmentVariableError(ValueError):
    """Raised when one or more required environment variables are missing."""

# This function loads Databricks environment variables and returns a typed config object.
def get_databricks_config(validate: bool = True) -> DatabricksConfig:
    """Load Databricks environment variables and return a typed config object."""
    load_dotenv()

    token = os.environ.get("DATABRICKS_TOKEN", "")
    host = os.environ.get("DATABRICKS_HOST", "")
    model = os.environ.get("DATABRICKS_MODEL", "")

    # If validate is True, check for missing variables and raise an error if any are not set.
    if validate:
        missing = [
            name
            for name, value in {
                "DATABRICKS_TOKEN": token,
                "DATABRICKS_HOST": host,
                "DATABRICKS_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise MissingEnvironmentVariableError(
                f"Missing required environment variable(s): {missing_text}"
            )

    return DatabricksConfig(token=token, host=host, endpoint=model)


def create_databricks_client(config: DatabricksConfig) -> openai.OpenAI:
    """Create an OpenAI client configured for Databricks model serving endpoints."""
    llm = ChatOpenAI(
        model=config.endpoint,
        api_key=config.token,
        base_url=f"{config.host}/serving-endpoints",
        temperature=0,
    )
    llm_noreason = ChatOpenAI(
        model=config.endpoint,
        api_key=config.token,
        base_url=f"{config.host}/serving-endpoints",
        reasoning_effort="none",
        temperature=0,
    )
    databricks_embeddings = OpenAIEmbeddings(
        model="databricks-gte-large-en",
        api_key=config.token,
        base_url=f"{config.host}/serving-endpoints",
        check_embedding_ctx_length=False
    )

    return llm, llm_noreason, databricks_embeddings

def get_tool_agent_instance(llm, tools):
    """Create an agent instance with the given LLM and tools."""
    agent = create_agent(llm=llm, tools=tools)
    return agent

def get_agent_instance(llm):
    """Create an agent instance with the given LLM and tools."""
    agent = create_agent(llm)
    return agent

def bootstrap_notebook(validate: bool = True):
    """Return notebook-ready variables: token, host, endpoint, and configured client."""
    config = get_databricks_config(validate=validate)
    llm, llm_noreason, databricks_embeddings = create_databricks_client(config)
    
    return config.token, config.host, config.endpoint, (llm, llm_noreason), databricks_embeddings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="pydantic")
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
        warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
    except Exception:
        pass

    pgvectordb_conn = "postgresql+psycopg://langchain:langchain!@localhost:5432/cs4603_vectordb"
    DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), databricks_embeddings = bootstrap_notebook()
