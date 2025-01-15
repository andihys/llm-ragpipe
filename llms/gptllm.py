# This is a script that allow you
# made an API call from your llm model.
#
# Is request to have set the .env file
# on which declare all the environment variables.

import logging
from typing import Optional
from config import azure_settings
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

def load_gpt_model(temperature: Optional[float] = None, top_p: Optional[float] = None) -> AzureChatOpenAI:
    """
    Loads the Azure OpenAI GPT model with specified configurations.
    """
    try:
        temp = temperature if temperature is not None else azure_settings.temperature
        top_p_value = top_p if top_p is not None else azure_settings.top_p

        llm = AzureChatOpenAI(
            azure_deployment=azure_settings.azure_deployment,
            api_version=azure_settings.azure_api_version,
            temperature=temp,
            top_p=top_p_value,
            timeout=30,
            max_retries=20,
            logprobs=False
        )
        logger.info("Successfully loaded GPT model.")
        return llm
    except Exception as e:
        logger.error(f"Failed to load GPT model: {e}", exc_info=True)
        raise

llm = load_gpt_model()

# for test the response of the model you can run this lines:
#
# response = llm.invoke("Hello, how are you?")
# print(response.content)
