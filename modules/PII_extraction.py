import json
import random
import time

import yaml
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger

from configs.global_params import *
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        config_home = Path(__file__).parent.parent / 'configs' / 'pii_categories'
        target_config_path = config_home / f'{pii_category}.yml'
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
                logger.warning(
                    f"[ConfigFormatError] Key detail config should be either str or dict. {type(key)} is not supported.")

        return keys_details, background

    def create_llm_instance(self, model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.95):
        with open(Path(__file__).parent.parent / 'configs' / 'configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return ChatOpenAI(temperature=temperature,
                          model=config['LLM'][model_name]['llm_params']['model_name'],
                          openai_api_key=config['LLM'][model_name]['llm_params']['api_key'],
                          openai_api_base=config['LLM'][model_name]['llm_params']['endpoint'])

    def unit_extract(self, pii_category, input_str):
        keys_details, background = self.load_config(pii_category)
        if not keys_details:
            logger.error("Failed to get key definitions.")
            return None
        extraction_prompt_template_path = Path(__file__).parent.parent / 'prompts' / 'pii_extraction.prompt'
        with open(extraction_prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        target_pii_class_string_part = []
        for key_detail in keys_details:
            target_pii_class_string_part.append(
                f'''- {key_detail["key_name"]}: {key_detail["key_description"] if key_detail["key_description"] else f"literal meaning of {key_detail['key_name']}"}''')
        parser = PydanticOutputParser(pydantic_object=PIIs)
        model_instance = self.create_llm_instance(temperature=round(random.uniform(0, 1), 2))

        prompt = template.format(timestamp=str(time.time()),
                                 background_str='\n'.join(background),
                                 example_section_str='',
                                 target_pii_class_string='\n'.join(target_pii_class_string_part),
                                 input_text=input_str,
                                 format_instruction=parser.get_format_instructions())
        # model_with_structure = model_instance.with_structured_output(MaskedContent)
        logger.debug(prompt)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        logger.success(answer)
        return json.loads(answer.model_dump_json()).get('piis', [])

    def extract(self, pii_category, input_str, votes=1):
        if votes < 1:
            logger.error("The number of votes must be at least 1.")
            return None

        def thread_task():
            return self.unit_extract(pii_category, input_str)

        vote_counter = defaultdict(Counter)

        with ThreadPoolExecutor(max_workers=min(5, votes)) as executor:
            future_tasks = [executor.submit(thread_task) for _ in range(votes)]

            for future in as_completed(future_tasks):
                try:
                    result = future.result()
                    if result:
                        for entry in result:
                            pii_content = entry.get("pii_content")
                            pii_class = entry.get("pii_class")
                            if pii_content and pii_class:
                                # Increment the vote count for the (pii_content, pii_class) pair
                                vote_counter[pii_content][pii_class] += 1
                            else:
                                logger.warning(f"Invalid entry structure: {entry}")
                    else:
                        logger.warning("One of the extraction tasks returned an empty result.")
                except Exception as e:
                    logger.error(f"Exception during extraction: {e}")

        if not vote_counter:
            logger.error("All extraction attempts failed.")
            return []
        logger.warning(vote_counter)
        final_results = []
        for pii_content, class_votes in vote_counter.items():
            most_voted_class, max_votes = class_votes.most_common(1)[0]
            if max_votes > max(1, votes // 3):
                final_results.append({"pii_content": pii_content, "pii_class": most_voted_class})

        logger.success(f"Final extracted PII: {final_results}")
        return final_results


if __name__ == "__main__":
    ins = PIIExtraction()
    # ins.extract('general', """COMPLAINT against 515 Restaurant, LLC, Jaybrien Estevez, Jay Grossman, Jessica Lantier, Union Turnpike Restaurant, LLC filing fee $ 405, receipt number ANYEDC-18649758 Was the Disclosure Statement on Civil Cover Sheet completed -NO,, filed by Emmi Liana Koutsidis. (Attachments: # 1 Appendix Document Preservation Hold Notice, # 2 Appendix Notice of Lien and Assignment, # 3 Appendix Anti-Retaliation Notice) (Troy, John) (Entered: 01/09/2025)""", 11)
    # ins.extract('internet_log',
    #             'Jan 21 03:39:56 PA-3410-01 1,2025/01/21 03:39:55,024109001378,TRAFFIC,drop,2562,2025/01/21 03:39:55,192.168.216.14,192.168.218.10,0.0.0.0,0.0.0.0,intrazone-server-deny,texl-eng\sysadm03,,not-applicable,vsys1,server,server,ae1.2,,Log_Forwarding_Profile,2025/01/21 03:39:55,0,1,55235,137,0,0,0x0,udp,deny,0,0,0,1,2025/01/21 03:39:55,0,any,,7458452573401715049,0x0,192.168.0.0-192.168.255.255,192.168.0.0-192.168.255.255,,1,0,policy-deny,0,0,0,0,,PA-3410-01,from-policy,,,0,,0,,N/A,0,0,0,0,5787f827-ec6d-44c7-9016-4e98cb265b8c,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,2025-01-21T03:39:56.771+08:00,,,unknown,unknown,unknown,1,,,not-applicable,no,no,0',
    #             11)
    ins.extract('internet_log', '''Jan 21 01:05:45 172.24.238.24 logver=702071577 timestamp=1737421544 devname="FWAWS104" devid="FGVMSLTM22000518" vd="root" date=2025-01-21 time=01:05:44 eventtime=1737392744510872814 tz="+0800" logid="0000000013" type="traffic" subtype="forward" level="notice" srcip=172.24.244.8 srcport=16122 srcintf="gwlb-a" srcintfrole="undefined" dstip=172.24.234.30 dstport=22 dstintf="gwlb-a" dstintfrole="undefined" srccountry="Reserved" dstcountry="Reserved" sessionid=3036557812 proto=6 vrf=20 action="accept" policyid=31575 policytype="security-policy" poluuid="8e67e89a-c77d-51ef-7406-58fa53396484" service="SSH" trandisp="noop" duration=3 sentbyte=2165 rcvdbyte=3129 sentpkt=12 rcvdpkt=13 appcat="unscanned"''',
                11)
