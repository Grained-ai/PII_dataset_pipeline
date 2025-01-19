import yaml
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger

from configs.global_params import *
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class PII(BaseModel):
    pii_content: str = Field(description='Content in the input text that is one of target PII classes.')
    pii_class: str = Field(description='Corresponding pii class')

class PIIs(BaseModel):
    piis: List[PII] = Field(description="List of piis and their corresponding class in the given content.")

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
        target_pii_class_string_part = []
        for key_detail in keys_details:
            target_pii_class_string_part.append(f'{key_detail["key_name"]}')
        parser = PydanticOutputParser(pydantic_object=PIIs)
        model_instance = self.create_llm_instance()

        prompt = template.format(background_str = '\n'.join(background),
                                 example_section_str = '',
                                 target_pii_class_string = '\n'.join(target_pii_class_string_part),
                                 input_text=input_str,
                                 format_instruction=parser.get_format_instructions())
        # model_with_structure = model_instance.with_structured_output(MaskedContent)
        logger.debug(prompt)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        logger.success(answer)
        return answer

    def extract(self, pii_category):
        pass

if __name__ == "__main__":
    ins = PIIExtraction()
    ins.extract_unit('general', """COMPLAINT against 515 Restaurant, LLC, Jaybrien Estevez, Jay Grossman, Jessica Lantier, Union Turnpike Restaurant, LLC filing fee $ 405, receipt number ANYEDC-18649758 Was the Disclosure Statement on Civil Cover Sheet completed -NO,, filed by Emmi Liana Koutsidis. (Attachments: # 1 Appendix Document Preservation Hold Notice, # 2 Appendix Notice of Lien and Assignment, # 3 Appendix Anti-Retaliation Notice) (Troy, John) (Entered: 01/09/2025)
""")
