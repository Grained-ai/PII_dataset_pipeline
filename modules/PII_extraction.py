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
from langchain_ollama import ChatOllama

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
                                     'optional_regex': [],
                                     'match_regex': []})
            elif isinstance(key, dict):
                key_name = list(key.keys())[0]
                key_detail = key[key_name]
                keys_details.append({'key_name': key_name,
                                     'key_description': key_detail.get('key_description'),
                                     'output_format_description': key_detail.get('output_format_description'),
                                     'examples': key_detail.get('examples'),
                                     'mandatory_regex': key_detail.get('mandatory_regex', []),
                                     'optional_regex': key_detail.get('optional_regex', []),
                                     'match_regex': key_detail.get('match_regex', [])})
            else:
                logger.warning(
                    f"[ConfigFormatError] Key detail config should be either str or dict. {type(key)} is not supported.")

        return keys_details, background

    @staticmethod
    def create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.95):
        # TODO: Refraction Needed.
        with open(Path(__file__).parent.parent / 'configs' / 'configs.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if model_name.startswith("OLLAMA"):
            return ChatOllama(temperature=temperature,
                              model=config['LLM'][model_name]['llm_params']['model_name'])
        else:
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
                f'''- {key_detail["key_name"]}: {key_detail["key_description"] if key_detail["key_description"] else f"{key_detail['key_name']}"} content in any format''')

        regex_extracted_piis = []
        for key_detail in keys_details:
            if key_detail.get('match_regex'):
                logger.warning(f"Will use REGEX to extract {key_detail['key_name']} First.")
            for regex_string in key_detail['match_regex']:
                search_res = re.findall(regex_string, input_str)
                if search_res:
                    logger.success(f"Found matched REGEX result: {search_res}")
                    regex_extracted_piis.extend([{'pii_content': i, 'pii_class': key_detail['key_name']} for i in search_res])

        parser = PydanticOutputParser(pydantic_object=PIIs)
        model_instance = self.create_llm_instance(temperature=round(random.uniform(0.5, 1), 2))

        prompt = template.format(timestamp=str(time.time()),
                                 background_str='\n'.join(background),
                                 example_section_str='',
                                 target_pii_class_string='\n'.join(target_pii_class_string_part),
                                 input_text=input_str,
                                 format_instruction=parser.get_format_instructions(),
                                 comments_str='' if not comment else '\n'.join(comment))
        # model_with_structure = model_instance.with_structured_output(MaskedContent)
        # logger.debug(prompt)
        try:
            res_content = model_instance.invoke(prompt)
            answer = parser.parse(res_content.content)
        except:
            return regex_extracted_piis
        # logger.success(answer)
        llm_extracted_piis = json.loads(answer.model_dump_json()).get('piis', [])
        return llm_extracted_piis+regex_extracted_piis

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
            if max_votes >= max(1, votes // 3):
                final_results.append({"pii_content": pii_content, "pii_class": most_voted_class})

        logger.success(f"[Extraction] Extracted PII: {final_results}")
        return final_results

    @staticmethod
    def mask_piis(input_str, extracted_piis):
        skipped_piis = []
        masked_content = input_str
        logger.debug(extracted_piis)
        for pii in extracted_piis:
            try:
                pii_content = pii.get('pii_content')
            except:
                logger.error(pii)
                continue
            if pii_content not in input_str:
                logger.warning(f'{pii_content} is not in input_str.')
                skipped_piis.append(pii)
                continue
            pii_label = pii['pii_class']
            # logger.debug(f"Previous content: \n{masked_content}")
            masked_content = re.sub(re.escape(pii_content), f'[{pii_label}]', masked_content)
            # logger.debug(f"Masked content: \n{masked_content}")
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
        model_instance = self.create_llm_instance(model_name='Zhipu_glm4_flash')
        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 target_pii_class_string='\n'.join(target_pii_class_string_part),
                                 masked_document=masked_content)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        comments.append(answer.comments)
        logger.info(f"[ValidateRecall]   {answer.if_remaining_piis}, {comments}")
        return answer.if_remaining_piis, comments

    def validate_accuracy(self, input_str, extracted_piis):
        pass

    def main(self, pii_category, input_str, votes=1):
        pii_extracted = []
        cur_run_pii_extracted = self.extract(pii_category, input_str, votes)
        masked_input = input_str
        # masked_input, skipped_piis = self.mask_piis(input_str, cur_run_pii_extracted)
        while 1:
            logger.info(f"Current RUN PII_extracted [{len(cur_run_pii_extracted)}]: {cur_run_pii_extracted}.")
            have_unmasked_piis, comments = self.validate_recall(pii_category, masked_input, cur_run_pii_extracted)
            # Used to filter PIIs
            masked_input, skipped_piis = self.mask_piis(masked_input, cur_run_pii_extracted)
            cur_run_pii_extracted = [i for i in cur_run_pii_extracted if i not in skipped_piis]
            pii_extracted += cur_run_pii_extracted
            logger.info(f"[Summary]     PII extracted [{len(pii_extracted)}] -> {pii_extracted}")
            if not have_unmasked_piis:
                logger.success(f"<{pii_category}> PII masked thoroughly.")
                logger.success(f"[MAIN]      Extracted piis {json.dumps(pii_extracted, indent=4, ensure_ascii=False)}")
                break
            else:
                logger.warning(f"Still have un-extracted PIIs. {comments}")
            cur_run_pii_extracted = self.unit_extract(pii_category, masked_input, comments)
            # masked_input, skipped_piis = self.mask_piis(masked_input, cur_run_pii_extracted)
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
             """2025-02-02T00:27:59+08:00 COMP6615226D {"time": "2025-02-01T16:27:59.3432132Z", "resourceId": "/tenants/bc0b541e-cf5f-48a5-a45d-7d041fe508a0/providers/Microsoft.aadiam", "operationName": "Sign-in activity", "operationVersion": "1.0", "category": "NonInteractiveUserSignInLogs", "tenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "resultType": "50097", "resultSignature": "None", "resultDescription": "Device Authentication Required - DeviceId -DeviceAltSecId claims are null OR no device corresponding to the device identifier exists.", "durationMs": 0, "callerIpAddress": "10.104.204.32", "correlationId": "3ad2b5b1-ef09-4c50-81ad-853aef57674d", "identity": "user_5f5d1039", "Level": 4, "location": "CN", "properties": {"id": "4d63c1c4-c95b-4d3c-8e0a-1271fd813e00", "createdDateTime": "2025-02-01T16:26:24.6955459+00:00", "userDisplayName": "user_5f5d1039", "userPrincipalName": "user_468aad99@domain_e9c4f53a.com", "userId": "6abee629-69f8-461c-9fab-71c4ff200168", "appId": "27922004-5251-4030-b22d-91ecd9a37ea4", "appDisplayName": "Outlook for iOS and Android", "ipAddress": "10.104.204.32", "status": {"errorCode": 50097, "failureReason": "Device Authentication Required - DeviceId -DeviceAltSecId claims are null OR no device corresponding to the device identifier exists.", "additionalDetails": "MFA requirement satisfied by claim in the token"}, "clientAppUsed": "Mobile Apps and Desktop clients", "userAgent": "Mozilla/5.0 (compatible; TESTAL 1.0) PKeyAuth/1.0", "deviceDetail": {"deviceId": "30a4051b-7be0-4754-a3b8-755991aa69e0", "displayName": "WS3A7C1DBA", "operatingSystem": "Ios 18.2.1", "trustType": "Azure AD registered"}, "location": {"city": "Chengdu", "state": "Sichuan", "countryOrRegion": "CN", "geoCoordinates": {"latitude": 30.653060913085938, "longitude": 104.06749725341797}}, "mfaDetail": {}, "correlationId": "3ad2b5b1-ef09-4c50-81ad-853aef57674d", "conditionalAccessStatus": "success", "appliedConditionalAccessPolicies": [{"id": "627ea70c-8186-40a5-aafb-d681edbc2d88", "displayName": "Block Access to O365 from Mobile Devices for All Users", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 7, "conditionsNotSatisfied": 0}, {"id": "b8944f3f-001d-4d7a-ad95-4b110f20a9f5", "displayName": "Block All Access from Untrusted IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "a72bcb04-325f-4428-8597-431bd40d390f", "displayName": "CA307 - Application  Access from IOS Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1039, "conditionsNotSatisfied": 0}, {"id": "a34fae20-bd1e-479e-95e8-23dbc7ad9064", "displayName": "Block Access to O365 from NON-COD IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "a2a977f6-8469-4a63-a26e-d33066160e17", "displayName": "Block MFA Registration from Untrusted IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "c061f6ef-16a9-4939-96ff-96b4de4b2de5", "displayName": "Block Exchange Online Access from BYO IOS", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 263, "conditionsNotSatisfied": 0}, {"id": "61e7500a-54bc-4a7b-9da8-d97ca911b485", "displayName": "Enable Exchange Online Access from Managed IOS Device", "enforcedGrantControls": ["RequireCompliantDevice", "RequireCompliantApp"], "enforcedSessionControls": ["SignInFrequency"], "result": "failure", "conditionsSatisfied": 23, "conditionsNotSatisfied": 0}, {"id": "fc6f39dc-08c0-45af-b52e-053e1a9e5dd8", "displayName": "AADC DirSync Account Restriction", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "31bd8466-c9a4-49da-9bfd-68bb5ea30c48", "displayName": "Enable MFA for Micorosft Azure Management - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "4f94a30e-8595-4522-8594-a7b38dcf7de7", "displayName": "Enable MFA for O365 - AAD Roles", "enforcedGrantControls": ["COD PA Training"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "907260ac-d47d-4cb1-a266-3f757b3c3d27", "displayName": "Enable MFA for CyberArk - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "91f9176a-3641-4c97-af5c-c02ee23bb741", "displayName": "Enable SIF for All Services from Intranet - 7 Days", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 3, "conditionsNotSatisfied": 1032}, {"id": "efddddee-44c8-435f-b661-60d11515a36b", "displayName": "Enable SIF for CyberArk - 12 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "476d40ea-446e-497b-913a-497e7c8464b2", "displayName": "Enable SIF for Cloud BTG Accounts - 4 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "e8a6b9ec-85c4-4948-afb8-544ce8202f89", "displayName": "Enable SIF for Critical Services - 12 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "986eb6ae-1169-4c2e-9194-a2317222a4d3", "displayName": "Block Access to Power Platform for All Users", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "179fc5f0-50e0-4594-9efb-fb07f8e75aba", "displayName": "CA301 - Register or Join Device \\u2013 MFA ", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "0fa2d843-684c-4487-92e1-8707576c26b9", "displayName": "CA306 - Application External Access from Linux Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "3d01b038-5495-4cbe-913b-08dd47898438", "displayName": "CA305 - Application External Access from Windows Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "14bab089-a460-4838-987c-1dfa7b0173c5", "displayName": "CA303 - Company Portal App External Login on Unmanaged Device", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "bcd46537-f715-4fa6-b9ca-0cd996f34571", "displayName": "CA304 - Application Access from OtherOS Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "3c38edf0-c14c-4b44-a85a-69118bb366ec", "displayName": "CA302 - Company Portal App External Login on Managed Device - MFA", "enforcedGrantControls": ["RequireCompliantDevice"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "332f46ad-ce38-4620-b757-a3445cb0bd1b", "displayName": "Block Restricted User For Citrix Access", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "8842877d-af5a-4e9c-9b65-405486a5a4bf", "displayName": "Enable MFA for Citrix Access", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "549eb90d-2f20-469b-a29b-4fb1d7204991", "displayName": "Enable SIF for Citrix Access - Every Time", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "c8e61b40-6a11-4d07-bc8a-6dcf5db87fe1", "displayName": "Enable Regular PA Training for PA users", "enforcedGrantControls": ["Mfa", "COD PA Training"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "cc67f465-32a1-4d6e-9901-7b95d7709396", "displayName": "Block Exchange Online Access from Browser of Managed IOS Device", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 7, "conditionsNotSatisfied": 16}, {"id": "0f959fe2-497a-4849-aad8-1a50abbc837e", "displayName": "Enable MFA for ReversingLabs Software Assurance Managed Service", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "e548b13e-f481-44ba-86e4-ee94f6fa5d49", "displayName": "Enable MFA for CNAPS2 - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "cd5e57d9-7085-4099-b7f4-1c5622aee6d3", "displayName": "Enable SIF for CNAPS2 - 4 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}], "authenticationContextClassReferences": [{"id": "urn:user:registersecurityinfo", "detail": "required"}, {"id": "urn:user:registerdevice", "detail": "previouslySatisfied"}], "originalRequestId": "4d63c1c4-c95b-4d3c-8e0a-1271fd813e00", "isInteractive": false, "tokenIssuerName": "", "tokenIssuerType": "AzureAD", "authenticationProcessingDetails": [{"key": "Legacy TLS (TLS 1.0, 1.1, 3DES)", "value": "False"}, {"key": "Is CAE Token", "value": "False"}], "networkLocationDetails": [], "clientCredentialType": "none", "processingTimeInMilliseconds": 133, "riskDetail": "none", "riskLevelAggregated": "none", "riskLevelDuringSignIn": "none", "riskState": "none", "riskEventTypes": [], "riskEventTypes_v2": [], "resourceDisplayName": "Microsoft Graph", "resourceId": "00000003-0000-0000-c000-000000000000", "resourceTenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "homeTenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "tenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "authenticationDetails": [], "authenticationRequirementPolicies": [], "sessionLifetimePolicies": [], "authenticationRequirement": "singleFactorAuthentication", "servicePrincipalId": "", "userType": "Member", "flaggedForReview": false, "isTenantRestricted": false, "autonomousSystemNumber": 4134, "crossTenantAccessType": "none", "privateLinkDetails": {}, "ssoExtensionVersion": "", "uniqueTokenIdentifier": "xMFjTVvJPE2OChJx_YE-AA", "authenticationStrengths": [], "incomingTokenType": "refreshToken", "authenticationProtocol": "none", "appServicePrincipalId": null, "resourceServicePrincipalId": "b636ca69-c274-4271-8ca7-1649a6ab81ba", "rngcStatus": 0, "signInTokenProtectionStatus": "none", "tokenProtectionStatusDetails": {"signInSessionStatus": "unbound", "signInSessionStatusCode": 1004}, "originalTransferMethod": "none", "isThroughGlobalSecureAccess": false, "conditionalAccessAudiences": [{"applicationId": "00000003-0000-0ff1-ce00-000000000000", "audienceReasons": "none"}, {"applicationId": "00000002-0000-0ff1-ce00-000000000000", "audienceReasons": "none"}, {"applicationId": "00000002-0000-0000-c000-000000000000", "audienceReasons": "none"}, {"applicationId": "ea890292-c8c8-4433-b5ea-b09d0668e1a6", "audienceReasons": "none"}], "sessionId": "811fe6e4-f4bb-4684-8978-5ce5de39f8b9", "resourceOwnerTenantId": "28bd4fd2-a6ec-43cc-a6ab-416b4ef214f3"}}
2025-02-01T23:30:33+08:00 COMPE3FDB04F <Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Microsoft-Windows-Security-Auditing' Guid='{54849625-5478-4994-a5ba-3e3b0328c30d}'/><EventID>4625</EventID><Version>0</Version><Level>0</Level><Task>12544</Task><Opcode>0</Opcode><Keywords>0x8010000000000000</Keywords><TimeCreated SystemTime='2025-02-01T15:30:33.499045800Z'/><EventRecordID>6858529940</EventRecordID><Correlation/><Execution ProcessID='1376' ThreadID='12412'/><Channel>Security</Channel><Computer>COMPE3FDB04F</Computer><Security/></System><EventData><Data Name='SubjectUserSid'>NT AUTHORITY\\\\SYSTEM</Data><Data Name='SubjectUserName'>BJVMWCDC01$</Data><Data Name='SubjectDomainName'>COD</Data><Data Name='SubjectLogonId'>0x3e7</Data><Data Name='TargetUserSid'>NULL SID</Data><Data Name='TargetUserName'>ltm-bind</Data><Data Name='TargetDomainName'>COD</Data><Data Name='Status'>0xc000006d</Data><Data Name='FailureReason'>%%2313</Data><Data Name='SubStatus'>0xc000006a</Data><Data Name='LogonType'>3</Data><Data Name='LogonProcessName'>Advapi  </Data><Data Name='AuthenticationPackageName'>MICROSOFT_AUTHENTICATION_PACKAGE_V1_0</Data><Data Name='WorkstationName'>WS4CCADB6B</Data><Data Name='TransmittedServices'>-</Data><Data Name='LmPackageName'>-</Data><Data Name='KeyLength'>0</Data><Data Name='ProcessId'>0x560</Data><Data Name='ProcessName'>C:\\\\Windows\\\\System32\\\\lsass.exe</Data><Data Name='IpAddress'>10.134.159.249</Data><Data Name='IpPort'>38058</Data></EventData></Event>""",
             1)
