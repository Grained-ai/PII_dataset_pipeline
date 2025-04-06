import yaml
from pathlib import Path
from configs.global_params import *
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class LLMFactory:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMFactory, cls).__new__(cls)
            cls._instance.expense = {}
        return cls._instance

    @staticmethod
    def create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.95):
        with open(Path(__file__).parent.parent / 'configs' / 'configs.yaml', 'r') as f:
            config = yaml.safe_load(f)

        if model_name.startswith("OLLAMA"):
            return ChatOllama(
                temperature=temperature,
                model=config['LLM'][model_name]['llm_params']['model_name']
            )
        else:
            return ChatOpenAI(
                temperature=temperature,
                model=config['LLM'][model_name]['llm_params']['model_name'],
                openai_api_key=config['LLM'][model_name]['llm_params']['api_key'],
                openai_api_base=config['LLM'][model_name]['llm_params']['endpoint']
            )

if __name__ == "__main__":
    ins = LLMFactory()
    a = ins.create_llm_instance()
    print(a.predict("HI"))