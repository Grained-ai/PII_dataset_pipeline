import yaml
from pathlib import Path
from loguru import logger

from configs.global_params import *
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1

class PIIExtraction:
    def __init__(self):
        pass

    @staticmethod
    def parse_extraction_rule_configs(config_file_path):
        """

        :param config_file_path:
        :return: keys_raw, backgrounds
        """
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
        return config_data.get('keys', []), config_data.get('background', [])

    def load_config(self, pii_category):
        config_home = Path(__file__).parent.parent/'configs'/'pii_categories'
        target_config_path = config_home/f'{pii_category}.yml'
        if not target_config_path.exists():
            logger.error(f"PII_Category: {pii_category} doesnt exist. Please initialize the config file first.")
            return None, None
        keys, background = self.parse_extraction_rule_configs(target_config_path)
        keys_details = []
        for key in keys:
            if isinstance(key, str):
                keys_details.append({'key_name': key,
                                     'key_description': None,
                                     'output_format_description': None,
                                     'examples': None,
                                     'mandatory_regex': [],
                                     'optional_regex': []})
            elif isinstance(key, dict):
                key_name = list(key.keys())[0]
                key_detail = key[key_name]
                keys_details.append({'key_name': key_name,
                                     'key_description': key_detail.get('key_description'),
                                     'output_format_description': key_detail.get('output_format_description'),
                                     'examples': key_detail.get('examples'),
                                     'mandatory_regex': key_detail.get('mandatory_regex', []),
                                     'optional_regex': key_detail.get('optional_regex', [])})
            else:
                logger.warning(f"[ConfigFormatError] Key detail config should be either str or dict. {type(key)} is not supported.")

        return keys_details, background

    def create_llm_instance(self, model_name=DEFAULT_LLM_MODEL_NAME):
        with open(Path(__file__).parent.parent / 'configs' / 'configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return ChatOpenAI(temperature=0.95,
                          model=config['LLM'][model_name]['llm_params']['model_name'],
                          openai_api_key=config['LLM'][model_name]['llm_params']['api_key'],
                          openai_api_base=config['LLM'][model_name]['llm_params']['endpoint'])

    def extract_unit(self, pii_category, input_str):
        keys_details, background = self.load_config(pii_category)
        if not keys_details:
            logger.error("Failed to get key definitions.")
            return None
        extraction_prompt_template_path = Path(__file__).parent.parent/'prompts'/'pii_extraction.prompt'
        with open(extraction_prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        prompt = template.format()

    def extract(self, pii_category):
        pass

if __name__ == "__main__":
    ins = PIIExtraction()
    i = ins.create_llm_instance()
    print("HERE")
