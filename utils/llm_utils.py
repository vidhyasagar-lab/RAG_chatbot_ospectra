import os
from langchain_openai import AzureChatOpenAI
from utils.env_utils import get_env

llm = AzureChatOpenAI(
    deployment_name=get_env("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=get_env("AZURE_OPENAI_API_KEY"),
    api_version=get_env("AZURE_OPENAI_API_VERSION"),  # ✅ REQUIRED
    temperature=0,
)
