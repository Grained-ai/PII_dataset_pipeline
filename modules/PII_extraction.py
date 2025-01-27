import json
import random
import re
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


class RecallValidationResult(BaseModel):
    if_remaining_piis: bool = Field(description='If there are still remaining not masked piis. True means still '
                                                'remaining. False means already masked all.')
    comments: str = Field(
        description="If there are still remaining piis not masked, name them out. If already pii free, return your comment.")


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
        # TODO: Refraction Needed.
        with open(Path(__file__).parent.parent / 'configs' / 'configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return ChatOpenAI(temperature=temperature,
                          model=config['LLM'][model_name]['llm_params']['model_name'],
                          openai_api_key=config['LLM'][model_name]['llm_params']['api_key'],
                          openai_api_base=config['LLM'][model_name]['llm_params']['endpoint'])

    def unit_extract(self, pii_category, input_str, comment=None):
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
                                 format_instruction=parser.get_format_instructions(),
                                 comments_str='' if not comment else '\n'.join(comment))
        # model_with_structure = model_instance.with_structured_output(MaskedContent)
        # logger.debug(prompt)
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

    @staticmethod
    def mask_piis(input_str, extracted_piis):
        skipped_piis = []
        masked_content = input_str
        for pii in extracted_piis:
            pii_content = pii['pii_content']
            if pii_content not in input_str:
                logger.warning(f'{pii_content} is not in input_str.')
                skipped_piis.append(pii)
                continue
            pii_label = pii['pii_class']
            logger.debug(f"Previous content: \n{masked_content}")
            masked_content = re.sub(pii_content, f'[{pii_label}]', masked_content)
            logger.debug(f"Masked content: \n{masked_content}")
        return masked_content, skipped_piis

    def validate_recall(self, pii_category, input_str: str, extracted_piis):
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'pii_extraction_recall_validation.prompt'
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
        keys_details, background = self.load_config(pii_category)
        target_pii_class_string_part = []
        for key_detail in keys_details:
            target_pii_class_string_part.append(
                f'''- {key_detail["key_name"]}: {key_detail["key_description"] if key_detail["key_description"] else f"literal meaning of {key_detail['key_name']}"}''')

        masked_content, skipped_piis = self.mask_piis(input_str, extracted_piis)
        comments = [f'pii: <{i["pii_content"]}> is  not in input_str' for i in skipped_piis]
        logger.info(f"Starts to validate if pii contained in \n{masked_content}")

        parser = PydanticOutputParser(pydantic_object=RecallValidationResult)
        model_instance = self.create_llm_instance()
        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 target_pii_class_string='\n'.join(target_pii_class_string_part),
                                 masked_document=masked_content)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        comments.append(answer.comments)
        return answer.if_remaining_piis, comments

    def validate_accuracy(self, input_str, extracted_piis):
        pass

    def main(self, pii_category, input_str, votes=1):
        pii_extracted = ins.extract(pii_category, input_str, votes)
        while 1:
            logger.info(f"Current PII_extracted [{len(pii_extracted)}]: {pii_extracted}.")
            have_unmasked_piis, comments = ins.validate_recall(pii_category, input_str, pii_extracted)
            if not have_unmasked_piis:
                logger.success(f"<{pii_category}> PII masked thoroughly.")
                break
            masked_input, skipped_piis = self.mask_piis(input_str, pii_category)
            pii_extracted = [i for i in pii_extracted if i not in skipped_piis]
            logger.info(f"PII extracted [{len(pii_extracted)}] -> {pii_extracted}")
            new_pii_extracted = self.unit_extract(pii_category, masked_input, comments)
            pii_extracted = pii_extracted + new_pii_extracted
        return pii_extracted

if __name__ == "__main__":
    ins = PIIExtraction()
    # ins.extract('general', """COMPLAINT against 515 Restaurant, LLC, Jaybrien Estevez, Jay Grossman, Jessica Lantier, Union Turnpike Restaurant, LLC filing fee $ 405, receipt number ANYEDC-18649758 Was the Disclosure Statement on Civil Cover Sheet completed -NO,, filed by Emmi Liana Koutsidis. (Attachments: # 1 Appendix Document Preservation Hold Notice, # 2 Appendix Notice of Lien and Assignment, # 3 Appendix Anti-Retaliation Notice) (Troy, John) (Entered: 01/09/2025)""", 11)
    # ins.extract('internet_log',
    #             'Jan 21 03:39:56 PA-3410-01 1,2025/01/21 03:39:55,024109001378,TRAFFIC,drop,2562,2025/01/21 03:39:55,192.168.216.14,192.168.218.10,0.0.0.0,0.0.0.0,intrazone-server-deny,texl-eng\sysadm03,,not-applicable,vsys1,server,server,ae1.2,,Log_Forwarding_Profile,2025/01/21 03:39:55,0,1,55235,137,0,0,0x0,udp,deny,0,0,0,1,2025/01/21 03:39:55,0,any,,7458452573401715049,0x0,192.168.0.0-192.168.255.255,192.168.0.0-192.168.255.255,,1,0,policy-deny,0,0,0,0,,PA-3410-01,from-policy,,,0,,0,,N/A,0,0,0,0,5787f827-ec6d-44c7-9016-4e98cb265b8c,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,2025-01-21T03:39:56.771+08:00,,,unknown,unknown,unknown,1,,,not-applicable,no,no,0',
    #             11)
    sample = """5.0/5.0
    By Holly on Jan 7, 2025
    Good clumping, low dust, no odor, NO FRAGRANCE!
    I've been looking for a new litter, because although it claims to be 99.9% dust free, the one I've been using for a couple of years leaves dust everywhere. Kept using it, because the cats like it, but fed up with the dust and mess. I used to use Tidy Cats a while back, but had switched to a cheaper brand. Took a chance on this and it covers all the bases: it clumps well and is easy to clean, there is truly no extra dust when I pour it and I'm not seeing dust tracked everywhere, odor control is good, and bonus, there is NO FRAGRANCE. I'm really sensitive to perfumes, and there are a lot of good litters that I can't use jut because of the scent. And most importantly, the kitties like it. They have actually refused other brands and litter types before. (I may have spoiled them :-/ ) Happy with this and making the switch back Tidy Cats, the OG.
    4Likes Report

    1.0/5.0
    By Lynn on Jan 9, 2025
    Good luck opening!
    To say this was a challenge to open is an understatement. After a half hour wrestling with this I opted to use a hammer. I kid you not, nor do I exaggerate. The seriously need to rethink the design on this. Never again! Which brings me to the little. No dust, which is wonderful. However removing the clumps from the pee is not very easy. It's nearly the consistency of cement. I had to purchase a, stainless steel scoop but it was still quite difficult. I only used this for 1 week. I purchased this as well as a bag I have opted to change litter over to another brand . I'm very disappointed in Tidy Cat and won't be recommending it to anyone.
    2LikesReport

    2.0/5.0
    By Katie on Jan 14, 2025
    Litter everywhere
    This isn't the first time I ordered one of these pails and had it delivered with litter strewn all over the box... but this was probably the worst. I always fear this ordering the pails for shipping, and everytime it happens I go back to buying them in store, until I can't make it in and use chewy. Just to regret it everytime. Order bags or the clear plastic totes for possibly better luck, otherwise just buy somewhere in store yourself.

    1LikeReport

    5.0/5.0
    By JaeBird on Jan 19, 2025
    Odor control
    I'd been using another, slightly cheaper brand in the litter robot for a few years. Tidy cats works so much better! Much less dust also. Love the refill bag(only have found at Chewy) - although the price difference is disappointingly negligible. Will go back to buckets when I have more room as cleaned out they are great for storage. Also love that I can dictate to FedEx to leave it at my front door so I don't have to carry it up stairs.
    0LikesReport

    1.0/5.0
    By Daniel on Jan 6, 2025
    Bad delivery
    The delivery person left both 38 lb. packages at the wrong address. As someone who has a number of physical issues, ie; 67 years old, broken kneee, broken ankle, bad hip and back, I rely on the heavy packages being delivered properly. Climbing up and down 3 flights of stairs several times to collect these packages causes me severe physical pain. This is not the first time this has happened and I've raised the concern. If it happens once more, I'll take my thousands of dollars of annual business somewhere else. Someone please school your delivery company.
    0LikesReport

    1.0/5.0
    By Marty on Jan 3, 2025
    Used this brans for years, UNCENTED was Great BUT CAUSED HEALTH PROBLEMS!
    After years of use my cat started have health problems! SHe would throw up gray matter like a cement looking substance. Spent lots of money and changed foods and such. Finally, the VET after checking the substance found that it was FROM THE CAT LITTER, and it was setting up in her digestive track and hardening! We switched to tree shavings, and she quit throwing up, but it was too late for her digestive track, she went for over a month without pooping and quit eating and drinking along the way. The vet didn't help and she suffered. The POWDER from the product collects on the paws and they lick it!
    6LikesReport

    4.0/5.0
    By Megan on Jan 10, 2025
    No smell
    I’ve experimented with numerous cat litters, all in an effort to eliminate that persistent feline odor. With this particular brand, a simple scoop and a modest top-up suffice, and the smell is completely gone. However, it does produce some dust, and the charcoal particles tend to cling to my cats’ fur but since they receive regular baths, this isn’t a major issue. We have 7 cats currently.

    3LikesReport"""
    # res = ins.extract('general', sample,
    #             5)
    ins.main('internet_log',
             'Jan 21 03:39:56 PA-3410-01 1,2025/01/21 03:39:55,024109001378,TRAFFIC,drop,2562,2025/01/21 03:39:55,192.168.216.14,192.168.218.10,0.0.0.0,0.0.0.0,intrazone-server-deny,texl-eng\sysadm03,,not-applicable,vsys1,server,server,ae1.2,,Log_Forwarding_Profile,2025/01/21 03:39:55,0,1,55235,137,0,0,0x0,udp,deny,0,0,0,1,2025/01/21 03:39:55,0,any,,7458452573401715049,0x0,192.168.0.0-192.168.255.255,192.168.0.0-192.168.255.255,,1,0,policy-deny,0,0,0,0,,PA-3410-01,from-policy,,,0,,0,,N/A,0,0,0,0,5787f827-ec6d-44c7-9016-4e98cb265b8c,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,2025-01-21T03:39:56.771+08:00,,,unknown,unknown,unknown,1,,,not-applicable,no,no,0',
             5)
