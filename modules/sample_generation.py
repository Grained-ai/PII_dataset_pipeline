import glob
import json
import os
import random
import time
import multiprocessing

import tqdm

from modules.pii_generators.person_pii_generator import PersonGenerator
from modules.pii_generators.diy_pii_generator import DIYPIIGenerator
from modules.PII_extraction import PIIExtraction
from modules.ocr_handler import OCRHandler
from loguru import logger
import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from configs.global_params import *
import concurrent.futures
from pathlib import Path
import yaml
from difflib import SequenceMatcher
from bs4 import BeautifulSoup


class RefineResult(BaseModel):
    refined_context: str = Field(
        description='Refined logically coherent Synthetic Content.You need to keep the format as it is')
    # refined_pii_mapping: dict = Field(description='Refined PII mapping in Dict form.')
    reason: str = Field(description='Reasons for the result')


class SynthesizedSample(BaseModel):
    synthesized_body: str = Field(description="Synthesized content generated.")
    used_piis: list = Field(description="List of used pii information in the synthesized content")


class SampleGeneration:
    def __init__(self):
        self.ocr_handler = OCRHandler()
        self.pii_extractor = PIIExtraction()

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
                              # seed = random.randint(10, 50),
                              model=config['LLM'][model_name]['llm_params']['model_name'],
                              openai_api_key=config['LLM'][model_name]['llm_params']['api_key'],
                              openai_api_base=config['LLM'][model_name]['llm_params']['endpoint'])

    @staticmethod
    def validate_template(template_body):
        person_ins = PersonGenerator()
        # Step 1: Extract direct placeholders in the form of {FirstName}, {FullName}, etc.
        direct_placeholders = re.findall(r'\{(.*?)\}', template_body)

        # Step 2: Extract method-call style placeholders in the form of [$$<method_name>(params)$$]
        method_placeholders = re.findall(r'\[\$\$(.*?)\$\$\]', template_body)

        # Step 3: Check if each direct placeholder has a corresponding generator in PIIGenerator enum
        missing_placeholders = []
        for placeholder in direct_placeholders:
            if not hasattr(person_ins, placeholder.split('_')[0]):
                missing_placeholders.append(placeholder)

        # Step 4: Check if each method placeholder has a corresponding method_name> method
        for method in method_placeholders:
            method_name = method.split('(')[0]  # Get the method name before the parentheses
            if not hasattr(DIYPIIGenerator, method_name.upper()):
                missing_placeholders.append(f"[${{method}}] method missing: {method}")

        if missing_placeholders:
            logger.error(f"Missing placeholders or methods: {', '.join(missing_placeholders)}")
            return missing_placeholders
        else:
            logger.success("All placeholders and methods are valid.")
            return True

    def generate_sample_by_image(self, image_path: Path, batch_dir: Path):
        lines = self.ocr_handler.get_ocr_result_by_block(image_path, output_path=batch_dir)
        sample_content = '\n'.join(lines)
        seed_path = batch_dir / 'seed_content.txt'
        with open(seed_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        pii_extracted = self.pii_extractor.main(pii_category='general', input_str=sample_content, votes=5)
        pre_text, placeholder_maps, person_mapping = self.generate_sample_by_sample_p_extraction(sample_content,
                                                                                                 pii_extracted)
        sample_path = batch_dir / 'sample_content.txt'
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(pre_text)
        return pre_text, {}, None

    @staticmethod
    def sample_to_template(input_str, extracted_piis):
        """

        :param input_str:
        :param extracted_piis:
        :return: Template body, different instance count
        """
        skipped_piis = []
        masked_content = input_str
        # logger.debug(extracted_piis)
        pii_label_to_pii_content_mapping = {}
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
            if pii_label not in pii_label_to_pii_content_mapping:
                pii_label_to_pii_content_mapping[pii_label] = []
            pii_label_to_pii_content_mapping[pii_label].append(pii_content)
            masked_content = re.sub(rf'\b{re.escape(pii_content)}\b(?![^\[]*\$\$\])',
                                    f"[$$Reformated Person_{len(pii_label_to_pii_content_mapping[pii_label]) - 1}'s {pii_label} according to the the format of this {pii_label} example: <{pii_content}>.$$]",
                                    masked_content)
        instance_count = max(
            [len(pii_label_to_pii_content_mapping[i]) for i in pii_label_to_pii_content_mapping.keys()])
        return masked_content, instance_count, pii_label_to_pii_content_mapping

    @staticmethod
    def determine_input_type(input_body):
        if isinstance(input_body, Path) and input_body.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp",
                                                                          ".tiff", ".webp"}:
            return 'IMAGE'
        if re.search(r"\[\$\$(.*?)\$\$]", input_body):
            return "TEMPLATE"
        if re.search(r"\{\{(.*?)}}", input_body):
            return "TEMPLATE"
        return "SAMPLE"

    def generate_sample_by_template(self, template_body, sample_count=1):
        """
        :param template_body:
        :param sample_count:
        :return: generated content
        """
        seed_content = None
        validation_res = self.validate_template(template_body)
        if not validation_res is True:
            logger.error(f"[Validation] Failed. {validation_res}")
            return seed_content, template_body, None

        def process_template():
            # Extract placeholders with optional underscores for multiple people
            direct_placeholders = re.findall(r'\{(\w+?)(?:_(\d+))?}', template_body)
            method_placeholders = re.findall(r'\[\$\$(.*?)\$\$]', template_body)

            # Determine the number of distinct PersonGenerator instances needed
            person_mapping = {}
            for placeholder, index in direct_placeholders:
                if index not in person_mapping:
                    person_mapping[index] = PersonGenerator()

            # Prepare mappings for direct placeholders
            sample_data = {}
            for placeholder, index in direct_placeholders:
                person_instance = person_mapping[index]  # Use the correct person instance
                if hasattr(person_instance, placeholder):
                    sample_data[f"{placeholder}_{index}" if index else placeholder] = getattr(person_instance,
                                                                                              placeholder)

            # Replace method placeholders with generated values
            for method in method_placeholders:
                if method.startswith('int'):
                    match = re.match(r'int\((\d+),\s*(\d+)\)', method)
                    if match:
                        min_val, max_val = map(int, match.groups())
                        generator = DIYPIIGenerator(DIYPIIGenerator.INT)
                        sample_data[method] = generator.generate(min_val, max_val)
                elif method.startswith('float'):
                    match = re.match(r'float\((\d+\.\d+),\s*(\d+\.\d+)\)', method)
                    if match:
                        min_val, max_val = map(float, match.groups())
                        generator = DIYPIIGenerator(DIYPIIGenerator.FLOAT)
                        sample_data[method] = generator.generate(min_val, max_val)
                elif method.startswith('list'):
                    match = re.match(r'list\((.*?)\)', method)
                    if match:
                        options = list(eval(match.group(1)))
                        generator = DIYPIIGenerator(DIYPIIGenerator.LIST)
                        sample_data[method] = generator.generate(options)
                elif method.startswith('prompt'):
                    match = re.match(r'prompt\((.*?)\)', method)
                    if match:
                        prompt = match.group(1)
                        generator = DIYPIIGenerator(DIYPIIGenerator.PROMPT)
                        sample_data[method] = generator.generate(prompt)

            # Replace all placeholders in the template
            filled_template = template_body
            for placeholder, value in sample_data.items():
                filled_template = filled_template.replace(f'{{{placeholder}}}', str(value))
                filled_template = filled_template.replace(f'[$${placeholder}$$]', str(value))

            logger.success(f"Processed template:\n {filled_template}")

            # Refine sample
            content, reason = self.fill_template_by_llm(filled_template, person_mapping)
            logger.success(f"Content Generated: \n{content}")
            return content

        # 并发执行 sample_count 次
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda _: process_template(), range(sample_count)))

        return seed_content, template_body, results

    # Deprecated
    def generate_sample_by_sample(self, sample_text):
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'sample_to_sample.prompt'
        person = PersonGenerator()
        my_info_string = person.summary_text()
        my_info_dict = person.summary()
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        parser = PydanticOutputParser(pydantic_object=SynthesizedSample)
        model_instance = self.create_llm_instance()

        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 real_sample_content=sample_text,
                                 my_info_string=my_info_string)
        res_content = model_instance.invoke(prompt)
        answer = parser.parse(res_content.content)
        logger.success(answer.synthesized_body)
        return answer.synthesized_body, answer.used_piis, my_info_dict

    def generate_sample_by_sample_p_extraction(self, sample_text):
        pii_extracted = self.pii_extractor.main(pii_category='general', input_str=sample_text, votes=5)
        template, instance_count, pii_label_to_pii_content_mapping = self.sample_to_template(sample_text,
                                                                                             extracted_piis=pii_extracted)
        logger.info(f"Generated template is: \n{template}")
        # Refine sample
        person_mapping = {str(i): PersonGenerator() for i in range(instance_count)}
        content, reason = self.fill_template_by_llm(template, person_mapping)
        logger.success(f"Content Generated: \n{content}")
        logger.success(f"\nComments: {reason}")
        # Extract filled items
        # mapping = self.extract_filled_content(sample_text, pii_label_to_pii_content_mapping, content)
        # logger.success(f"Filled content: {json.dumps(mapping, indent=2, ensure_ascii=False)}")
        # return content, mapping, reason
        return sample_text, template, content
        # sample, used_piis, my_info_dict = self.generate_sample_by_template(template_body=template)
        # return sample, used_piis, my_info_dict

    # @staticmethod
    # def util_align_generated_content_with_seed_content(seed_content, generated_content):
    #     seed_lines = seed_content.split('\n')
    #     generated_lines = generated_content.split('\n')
    #
    #     if len(seed_lines) == len(generated_lines):
    #         return seed_lines, generated_lines
    #
    #     aligned_seed = []
    #     aligned_generated = []
    #
    #     seed_ptr = 0
    #     generated_ptr = 0
    #
    #     while seed_ptr == generated_ptr and seed_ptr < len(seed_lines) and generated_ptr < len(generated_lines):
    #         if seed_lines[seed_ptr] == generated_lines[generated_ptr]:
    #             if seed_ptr == generated_ptr:
    #                 aligned_seed.append(seed_lines[seed_ptr])
    #                 aligned_generated.append(generated_lines[generated_ptr])
    #             else:
    #                 aligned_seed.append(seed_lines[seed_ptr])
    #                 aligned_generated.extend([''] * (seed_ptr - generated_ptr))
    #             seed_ptr += 1
    #             generated_ptr += 1
    #             continue
    #         aligned_seed.append(seed_lines[seed_ptr])
    #         seed_ptr += 1
    #
    #     return aligned_seed, aligned_generated

    def generate_sample_by_html_p_extraction(self, sample_text, storage_name=None):
        if storage_name is None:
            storage_name = str(int(time.time() * 1000))
        logger.info(f"Name: {storage_name}")
        if (Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}.html').exists():
            logger.success(f"{storage_name} Already finished.")
            return
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(sample_text, 'lxml')

        # 提取所有文本节点，排除纯空白或仅包含空白字符的文本
        texts = [element.get_text(strip=True) for element in soup.find_all(text=True) if element.get_text(strip=True)]
        if not texts:
            logger.error(f"Template {storage_name} has no text")
            return
        text_all = '\n'.join(texts)
        pii_extracted = self.pii_extractor.main(pii_category='ecommerce', input_str=text_all, votes=5)
        try:
            template, instance_count, pii_label_to_pii_content_mapping = self.sample_to_template(text_all,
                                                                                                 extracted_piis=pii_extracted)
            logger.info(f"Generated template is: \n{template}")

        except Exception as e:
            logger.error(f"Failed to sample2template. \n{e}")
            return
        # Refine sample
        person_mapping = {str(i): PersonGenerator() for i in range(instance_count)}
        person_mapping_store = {i: person_mapping[i].summary_serializable() for i in person_mapping.keys()}
        line_mapping = {}
        generated_content = None
        retry_notice = []
        for attempt in range(5):
            try:
                content, reason = self.fill_template_by_llm(template, person_mapping, truncated_version=2,
                                                            retry_notice=retry_notice)
            except Exception as e:
                logger.error(f"Failed to fill by LLM: {e}")
                retry_notice.append(str(e))
                continue
            content = re.sub(r"\n+", "\n", content)
            logger.info(f"Content Generated: \n{content}")
            logger.info(f"\nComments: {reason}")
            text_all = re.sub(r"\n+", "\n", text_all)
            logger.info(f"Prev content: \n{text_all}")
            if len(list(text_all.split('\n'))) != len(list(content.split('\n'))):
                len_text_all = len(list(text_all.split('\n')))
                len_content = len(list(content.split('\n')))
                # if abs(len_text_all - len_content) >= len_content*0.2:
                logger.warning(
                    f"Generated content line count doesnt match {len_text_all} != {len_content} . Generate again.")
                continue
            for idx, line in enumerate(list(text_all.split('\n'))):
                if line == content.split('\n')[idx]:
                    continue
                line_mapping[line] = content.split('\n')[idx]
            generated_content = content
            break
        if generated_content:
            logger.success(f"Content Generated: \n{generated_content}")
            html_sample = sample_text
            for line in line_mapping:
                # html_sample = re.sub(line, line_mapping[line], html_sample)
                html_sample = html_sample.replace(line, line_mapping[line])
            filled_contents = self.extract_filled_content(template, generated_content)
            os.makedirs(Path(__file__).parent / 'ecommerce_outs', exist_ok=True)
            with open(Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}_filled.json', 'w') as f:
                json.dump(filled_contents, f, ensure_ascii=False, indent=2)
            with open(Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}.html', 'w') as f:
                f.write(html_sample)
            with open(Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}.json', 'w') as f:
                json.dump(person_mapping_store, f, ensure_ascii=False, indent=2)

            logger.success(f"Sample stored at {Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}.html'}")
        else:
            logger.error(f"Failed to generate: {storage_name}")

    def fill_template_by_llm(self, pre_text, person_mapping, truncated_version=0, retry_notice=None):
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'template_fill_by_LLM.prompt'
        person_mapping_part = ["Here are person and corresponding personal Information related to the context."]
        for person_id in person_mapping:
            person_notation = "main_person" if person_id == '' else f"person_{person_id}"
            person_info = person_mapping[person_id].summary()
            person_mapping_part.append(f"{person_notation} name is: {person_info['FullName']}: \n{str(person_info)}")
        person_mapping_string = '\n'.join(person_mapping_part)

        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Force truncated mode
        if len(pre_text.split('\n')) >= 50:
            logger.warning("Forced to use truncated version 2")
            truncated_version = 2

        start_append_parts = []
        end_append_parts = []
        if truncated_version == 1:
            # Process pre_text
            start_line = 0
            end_line = 0
            pre_text_lines = pre_text.split('\n')
            for idx, line in enumerate(pre_text_lines):
                if re.search(r'\$\$(.*?)\$\$', line):
                    logger.info(f"Start line index = {idx}")
                    start_line = idx
                    break
            for idx, line in enumerate(pre_text_lines[::-1]):
                if re.search(r'\$\$(.*?)\$\$', line):
                    logger.info(f"Start line index = -{idx}")
                    end_line = -idx
                    break
            pre_text_truncated = pre_text_lines.copy()

            if end_line != 0:
                pre_text_truncated = pre_text_truncated[:end_line]
                end_append_parts = pre_text_lines[end_line:]
            if start_line != 0:
                pre_text_truncated = pre_text_truncated[start_line:]
                start_append_parts = pre_text_lines[:start_line]
            logger.debug(f"Initial lines: {len(pre_text_lines)} => {len(pre_text_truncated)}")
        elif truncated_version == 2:
            pre_text_lines = pre_text.split('\n')
            pre_text_truncated = []
            for line in pre_text_lines:
                if re.search(r'\$\$(.*?)\$\$', line):
                    pre_text_truncated.append(line)
        else:
            pre_text_truncated = pre_text.split('\n')

        parser = PydanticOutputParser(pydantic_object=RefineResult)
        model_instance = self.create_llm_instance(temperature=0, model_name='zhipu_glm4_plus')
        # model_instance = self.create_llm_instance(temperature=0)

        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 # pii_mappings=pii_mappings,
                                 document_content='\n'.join(pre_text_truncated),
                                 person_mapping_string=person_mapping_string,
                                 retry_notice_str='' if not retry_notice else "\n".join(
                                     ["- " + i for i in list(set(retry_notice))]))
        # logger.debug(prompt)
        res_content = model_instance.invoke(prompt)
        logger.warning(f"{res_content.model_dump_json()}")
        answer = parser.parse(res_content.content)
        logger.success(answer)
        if re.search(r'\$\$(.*?)\$\$', answer.refined_context):
            raise Exception("Not yet filled template. Template still have [$$<prompt>$$] placeholders.")
        # Align outs
        if truncated_version == 1:
            refined_context_parts = start_append_parts + answer.refined_context.split('\n') + end_append_parts
        elif truncated_version == 2:
            refined_context_parts = []
            to_refine_refined_context_parts = answer.refined_context.split('\n')
            cur_todo_idx = 0
            total_todo_count = 0
            for line in pre_text.split('\n'):
                if re.search(r'\$\$(.*?)\$\$', line):
                    total_todo_count += 1
            if total_todo_count != len(to_refine_refined_context_parts):
                logger.error(f"Current Row counts: {len(to_refine_refined_context_parts)}")
                raise Exception(
                    f"There are {total_todo_count} lines(\\n) in seed content. Generated content should also have {total_todo_count} lines.")
            for line in pre_text.split('\n'):
                if re.search(r'\$\$(.*?)\$\$', line):
                    refined_context_parts.append(to_refine_refined_context_parts[cur_todo_idx])
                    cur_todo_idx += 1
                else:
                    refined_context_parts.append(line)
        else:
            refined_context_parts = answer.refined_context.split('\n')
        return '\n'.join(refined_context_parts), answer.reason

    @staticmethod
    def extract_filled_content(template, generated_content):
        template_lines = template.split('\n')
        generated_content_lines = generated_content.split('\n')
        if len(template_lines) != len(generated_content_lines):
            logger.error(
                f"Failed to extract filled content. Template Lines: {len(template_lines)}. Content Lines: {len(generated_content_lines)}")
            return
        placeholder_res_list = []
        for template_line, generated_line in zip(template_lines, generated_content_lines):
            direct_placeholders = re.findall(r'(\{.*?\})', template_line)
            method_placeholders = re.findall(r'(\[\$\$.*?\$\$\])', template_line)
            placeholders = direct_placeholders + method_placeholders
            if placeholders:
                logger.info(f'Extracting: {placeholders}')
                for placeholder in placeholders:
                    try:
                        search_res = re.search(template_line.replace(placeholder, '(.*)'), generated_line)
                        if search_res:
                            placeholder_res = search_res.group(1)
                            placeholder_res_list.append({placeholder: placeholder_res})
                            logger.success({placeholder: placeholder_res})
                    except Exception as e:
                        logger.error(e)
        return placeholder_res_list


    def main(self, input_body, batch_dir=None, sample_count=1):
        if not batch_dir:
            batch_dir = Path(__file__).parent / 'output' / str(time.time() * 1000000)
        if isinstance(batch_dir, str):
            batch_dir = Path(batch_dir)
        logger.info(f"STARTS TO GENERATE SAMPLE AT {batch_dir}")
        input_type = self.determine_input_type(input_body)
        logger.info(f"Input body is {input_type}")

        if input_type == "TEMPLATE":
            # content, mapping, reason = self.generate_sample_by_template(input_body)
            seed_document, template_body, synthetic_content = self.generate_sample_by_template(input_body,
                                                                                               sample_count=sample_count)
            # return content, mapping, reason
            return seed_document, template_body, synthetic_content
        elif input_type == "SAMPLE":
            # pre_text, placeholder_maps, person_mapping = self.generate_sample_by_sample_p_extraction(input_body)
            seed_document, template_body, synthetic_content = self.generate_sample_by_sample_p_extraction(input_body)
            # return pre_text, {}, None
            return seed_document, template_body, synthetic_content

        elif input_type == "IMAGE":
            # pre_text, placeholder_maps, person_mapping = self.generate_sample_by_image(input_body, batch_dir)
            seed_document, template_body, synthetic_content = self.generate_sample_by_image(input_body, batch_dir)
            # return pre_text, {}, None
            return seed_document, template_body, synthetic_content

        else:
            logger.error(f"Input type {input_type} not supported")
            return None, None, None


def process_file(file):
    with open(file, 'r') as f:
        data = f.read()
    st = time.time()
    ins = SampleGeneration()
    ins.generate_sample_by_html_p_extraction(data, Path(file).stem)
    elapsed_time = time.time() - st
    return file, elapsed_time  # 返回文件名和耗时，方便后续记录或统计

if __name__ == "__main__":
    # SAMPLE_TEMPLATE = """CERTIFICATE OF NATURALIZATION
    # Personal description of holder as of date of naturalization:
    # Date of birth: {DateOfBirth}
    # Sex: {Sex}
    # Height: [$$int(4,7)$$] feet [$$int(0,11)$$] inches
    # Marital status: [$$list('Married', 'Single')$$]
    # Country of former nationality: {Country}
    # USCIS Registration No. [$$prompt()$$] or {GeneralIDs}
    # I certify that the description given is true, and that the photograph affixed hereto is a likeness of me:
    # {FullName}
    # Be it known that, pursuant to an application filed with the Secretary of Homeland Security
    # at: {City}, {State}
    # The Secretary having found that:
    # {FullName}
    # Residing at:
    # {City}{State}
    # having complied in all respect with all of the applicable provisions of the naturalization laws of the United States, and having taken the oath of allegiance at a ceremony conducted by
    # US Citizenship And Immigration Services
    # at: {City}{State} on: {Date}
    # such person is admitted as a citizen of the United States of America"""
    # SAMPLE_TEMPLATE = """Are you a citizen of the United States of America? [$$list('yes', 'no')$$]
    # Will you be 18 years old on or before election day? [$$list('yes', 'no')$$]
    # If you checked "No" in response to either of these questions, do not complete form.
    # [$$list('Mr.', 'Miss', 'Mrs.', 'Ms.')$$] Last Name {LastName} First Name {FirstName} Middle Name(s) [$$prompt(mdnm)$$] [$$list('Jr', 'Sr', 'II', 'III', 'IV')$$]
    # Home Address {Address} Apt. Or Lot # [$$prompt()$$] City/Town {City} State {State} Zip Code {ZipCode}
    # Address Where You Get Your Mail If Different From Above {Address} City/Town {City} State {State} Zip Code {ZipCode}
    # Date of Birth {DateOfBirth}
    # Telephone Number (optional) {PhoneNumber}
    # ID Number - (See item 6 in the instructions for your state) {GeneralIDs}
    # Choice of Party (see item 7 in the instructions for your State) [$$list('Republican', 'Democratic', 'The Green', 'Libertarians', 'Constitution', 'Natural Law')$$]
    # Race or Ethnic Group (see item 8 in the instructions for your State) [$$list('White', 'Hispanic or Latino', 'Black', 'Asian', 'Native American', 'Pacific Islander')$$]
    # I have reviewed my state's instructions and I swear/affirm that:
    # I am a United States citizen
    # I meet the eligibility requirements of my state and subscribe to any oath required
    # The information I have provided is true to the best of my knowledge under penalty of perjury. If I have provided false information, I may be fined, imprisoned, or (if not a U.S. citizen) deported from or refused entry to the United States. {FullName} Date: {Date}"""
    #
    # SAMPLE_TEMPLATE = """Incident investigation report
    # This template is provided for example purposes. If you choose to use this template, make sure you customize it to your work and work site.Date and time of incident: {Date} [$$prompt('hour', 'minute')$$] [$$list('AM', 'PM')$$]
    # Incident location: {Address}
    # Date the incident was reported to OHS: <Indicate if not applicable> {Date}
    # Other parties involved in the incident: <Indicate if not applicable> [$$prompt('person(s)')$$]
    # Incident category: <Choose all that apply. Refer to Section 33 of the OHS Act for specifics.> [$$list('fatality', 'hospitalization', 'crane/derrick/hoist collapse', 'unplanned fire/explosion/flood', 'collapse/failure of structure or building', 'mine or mine site incidnet (Section 544 of the OHS Code)', 'radiation overexposure', 'potentially serious incident', 'other [$$prompt('electricity, asbestos')$$]]
    # Circumstances of injury, illness, incident or worker exposure<Follow the prompts below to describe the circumstances of the incident. Add or delete sections as needed. Do not include personal information (e.g., names, job titles, details of injury or illness) unless it is necessary and permitted by privacy law.>)Sequence of events <List what happened, in chronological order. Include visual aids such as sketches or diagrams if those help describe the incident.>
    # [$$prompt('Event 1 occurred')$$]
    # [$$prompt('Event 2 was induced')$$]
    # [$$prompt('Event 3 was the result')$$]
    # Work activities <Describe how many people were involved and in what capacity: for example, Worker One was doing task A; Worker Two was doing task B; Supervisor was doing task C.>
    # [$$prompt('Person 1 did thing A')$$]
    # [$$prompt('Person 2 did thing B)$$]
    # [$$prompt('Person 3 was doing thing C)$$]
    # Tools, materials, equipment <Include any relevant information: for example, condition, maintenance history, date last used, manufacturer’s specifications, safeguards, personal protective equipment.>
    # [$$prompt('Warranty A expired)$$]
    # [$$prompt('Product B expired')$$]
    # [$$prompt('Safeguard was not maintained')$$]
    # Work site conditions <Describe relevant conditions: for example, weather, harmful substances in use, noise, lighting, time of day, confined/restricted space, ergonomics.>
    # [$$prompt('The humidity was high')$$]
    # [$$prompt('The time was xx:xx pm')$$]
    # [$$prompt('The environment was dark')$$]
    # Organizational factors <Describe relevant systemic factors, such as communication methods, training for work site activities, safe work procedures, hazard assessment and control, supervisory requirements. >
    # [$$prompt('Walkie-talkies')$$]
    # [$$prompt('Safety cones were not in place')$$]
    # [$$prompt('Workers are required to shut off machinery before leaving site')$$]
    # Other circumstances <Describe any other circumstances relevant to the incident.>
    # [$$prompt('Person A was ill')$$]
    # [$$prompt('Person B is half blind)$$]
    # [$$prompt('Previous reports on xxx has been filed in the past)$$]
    # Circumstances: <List each identified circumstance that contributed to the incident. One per row. Add or delete rows as needed.> [$$prompt('Previous prompt A')$$]
    # Corrective action required: <List each identified circumstance that contributed to the incident. One per row. Add or delete rows as needed.> [$$prompt('Previous prompt A')$$]
    # Assigned to (position): [$$prompt('Site supervisor')$$]
    # Date completed: {Date}
    # Date report completed: {Date}
    # Date report provided to health and safety committee/representative/workers: <Indicate if not applicable.> {Date}"""
    #
    # SAMPLE_TEMPLATE = """{FullName}
    # {Address}
    # {City}{State}{ZipCode}
    # {Date}
    # [$$prompt(Custodian of Records/U.S. Army Corps of Engineers)$$]
    # [$$prompt(Records Supervisor)$$]
    # [$$prompt(CompanyName/Grained.AI)$$]
    # {StreetName}{StreetNumber}
    # {City}{State}{ZipCode}
    # Dear [$$prompt(Custodian of Records/U.S. Army Corps of Engineers)$$]:
    # Under the Alabama Open Records Law § 36-12-40 et seq., I am requesting an opportunity to inspect or obtain copies of public records that [$$prompt(related to the meeting minutes and agendas of the Montgomery City Council for all meetings held between January 1, 2024, and December 31, 2024. Specifically, I am seeking records that detail discussions or decisions regarding city zoning changes, infrastructure projects, or budget allocations for public works.)$$]
    # If there are any fees for searching or copying these records, please inform me if the cost will exceed $[$$int(0, 10000)$$].  However, I would also like to request a waiver of all fees in that the disclosure of the requested information is in the public interest and will contribute significantly to the public’s understanding of [$$prompt(Here, you can identify yourself as a representative of the news media if applicable and state that your request is related to news gathering purposes / how municipal infrastructure funding decisions are made and their impact on community development. This request is related to news gathering purposes as part of an investigative report for a local news publication to inform residents about transparency and accountability in government spending.)$$] This information is not being sought for commercial purposes.
    # The statute requires a response in a reasonable time period.  If access to the records I am requesting will take longer, please contact me with information about when I might expect copies or the ability to inspect the requested records.
    # If you deny any or all of this request, please cite each specific exemption you feel justifies the refusal to release the information and notify me of the appeal procedures available to me under the law.
    # Thank you for considering my request.
    # Sincerely,
    # {FullName}
    # {PhoneNumber}"""
    #
    # SAMPLE_TEMPLATE = """No. A[$$prompt(1234567)$$]
    # CERTIFICATE OF CITIZENSHIP
    # Personal description of holder as of date of issuance of this certificate.
    # Sex: [$$list('male', 'female')$$]
    # Height: [$$int(1, 8)$$] feet [$$int(0, 11)$$] inches
    # Marital Status: [$$list('married', 'not married')$$]
    # Country of birth: [$$prompt(China)$$]
    # I certify that the description above given is true, and that the photograph affixed hereto is a likeness of me
    #
    # Be it known that: {FullName} now residing at {Address} having applied to the Director of U.S. Citizenship and Immigration Services for a certificate of citizenship pursuant to Section 341 of the Immigration and Nationality Act, having proved to the satisfaction of the Director, that (s)he is now a citizen of the United States of America, became a citizen thereof on {Date} and is now in the United States.
    # Now Therefore, in pursuance of the authority contained in Section 341 of the Immigration and Nationality Act, this certificate of citizenship is issued this [$$list('1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st')$$] day of [$$list('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')$$] and the seal of the Department of Homeland Security affixed pursuant to statute 8 U.S.C § 1449
    # {FullName_2}"""
    SAMPLE_TEMPLATE = """FedEx
Commercial Invoice
Date of Export:{Date}
Shipper/Exporter (complete name and address): {FullName}{Address}
Country of Export: {Country}
Export References (Invoice No., PO No., etc.): [$$prompt(An invoice number format: INV-[$$int(100, 9999)$$])$$]
Recipient (complete name and address): {FullName_1}{Address_1}
Importer – if other than recipient (complete name and address):
Country of Manufacture: {Country_1}
Country of Ultimate Destination: {Country}
Federal Express International Air Waybill No.: [$$int(100000000000, 999999999999)$$]
Shipment Details
Number of Packages: [$$int(1, 8)$$]
Type of Packaging: [$$list('Boxes', 'Envelopes', 'Bags', 'Containers')$$]
Full Description of Goods: [$$prompt(Description of package content, e.g. Food products, Documents, Clothing, etc)$$]
Total Number of Packages: [$$prompt(same as Number of Packages)$$]
Currency: [$$prompt(Choose a currency the same as Country of Export)$$]
Customs Tariff Code (HS Code): [$$prompt(choose a code in the format: 0000.00, e.g. 8275.31)$$]
Weight: [$$int(1, 50)$$]
Total Weight: [$$int(1, 50)$$]
Declaration
I declare all the information contained in this invoice to be true and correct.
Signature of Shipper/Exporter (type name and title):
Date: {Date_1}
Check one:
[$$list('FOB', 'CIF')$$]
"""
    seed_content = """PANDORA\nView in\nbrowser\n.\nNEW\nCHARMS\nBRACELETS\nRINGS\nNECKLACES\nEARRINGS\nHello Angus,\nTHANK YOU FOR YOUR ORDER\nWe have received your order and are preparing it for despatch.\nExpress delivery orders should reach you in 1-2 working days. Standard delivery orders should reach you between 2-4 working days. Pandora\'s response to COVID-19:\nRead more\nORDER SUMMARY\nMy Spacer Micro Charm\nItem No: 798533C00\nSize: One size\nMetal: Sterling Silver\nQuantity: 1\nUnit price:\n£10.00\nORDER SUB-TOTAL:\n£10.00\nSHIPPING:\n£2.99\nDISCOUNT:\n£0.00\nTOTAL INCL VAT:\n£12.99\nVAT AMOUNT:\n£2.17\nPAYMENT METHOD:\nPAYPAL\nYOUR ADDRESSES\nSHIPPING INFORMATION\nAngus Samwell\nCitadel Investment Group Europe Ltd\nMoor House, 120 London Wall\nEC2Y 5ET,\n   \n    London\nGB\n75554751\nBILLING INFORMATION\nAngus Samwell\nCitadel Investment Group Europe Ltd\nMoor House, 120 London Wall\nEC2Y 5ET,\n   \n   London\nGB\n75554751\nHAVE A QUESTION ABOUT YOUR ORDER?\nLet us help you answer it. Browse our FAQs, send us an email, or speak to our Customer Care team.\n                                                                      We are not able to process exchanges at this time.\nMore information here to return your order.\nHELP IS HERE\nJOIN PANDORA CLUB\nAs a Pandora club member, you enjoy easy checkout, special treats and much more.\nBECOME A MEMBER\nWith love,\nPandora\nMY ACCOUNT\nPANDORA CLUB\nFAQS\nCONTACT US\nSTORE LOCATOR\nMY ACCOUNT\nPANDORA CLUB\nFAQS\nCONTACT\nSTORE LOCATOR\nSTANDARD DELIVERY\n£2.99\nFREE AND EASY\nUK RETURNS\nPRIVACY POLICY\nTERMS & CONDITIONS\nCOOKIE POLICY\nPRIVACY POLICY\nTERMS & CONDITIONS\nCOOKIE POLICY\nPlease note: It is not possible to return an online order in store. Exchange only. For full details of how to return your item for a refund, please visit our\nreturns\npage.\nPlease do not reply to this email. If you would like to contact us please\nclick here\n.\nTerms & Conditions apply.\nProducts purchased via the Pandora website will be sold to you by Pandora Jewellery UK Limited, registered in England and Wales under company number 06654012 whose registered address is at 33 George Street, London, W1U 3BH. VAT number GB213663917.\n©Pandora. All rights reserved.\nTERMS AND CONDITIONS\nThis website at the URL (uk.pandora.net/en) ("the Pandora website") is owned by Pandora A/S, a company registered in Denmark under Central Business Register number: 28505116, whose registered office is at HOVEDKONTOR, Havneholmen 17-19, DK-1561 Copenhagen V, Denmark ("Pandora A/S").\nProducts purchased via the Pandora website will be sold to you by Pandora Jewellery UK Limited, a company registered in England and Wales under company number 06654012 whose registered address is at 33 George Street, London, W1U 3BH ("Pandora"). Pandora\'s VAT number is GB213663917.\nIt is important that you read these terms and conditions ("Sales Terms") carefully before ordering any Products from the Pandora website. Your purchase of Products via the Pandora website will be governed by the Sale Terms. By ordering any Products, you agree to be bound by the Sale Terms.\n1. About These Sales Terms\nThese Sales Terms are applicable to all orders and purchases made via the Pandora website. Pandora may revise these Sales Terms from time to time. Any changes to the Sales Terms will apply on or after the date that the revised Sales Terms are published. You are advised to check this page from time to time to take notice of any changes, as they are binding on you.\nPandora offers products for sale via the Pandora website to consumer customers. Pandora reserves the right to not fulfil orders placed by non-consumers or by individuals who fail to comply with these terms.\nWith regard to purchases made by you via the Pandora website, Pandora will usually communicate with you electronically by email, using the email address provided to Pandora when you place an order or otherwise communicate with Pandora. You may still contact us using one of the methods referred to in\nsection 17 (Contact Details)\nbelow or by using the contact details on the Pandora website.\nIn using the Pandora website you agree to comply with\nPandora\'s Acceptable Use Policy\n.\nBEFORE PLACING AN ORDER FOR PRODUCTS YOU MUST READ AND AGREE TO BE BOUND BY THESE SALES TERMS.\nYOUR ATTENTION IS DRAWN IN PARTICULAR TO THE PROVISIONS OF\nSECTION 15 (LIABILITY)\n.\n                                              IF YOU FIND YOURSELF UNABLE TO AGREE TO THESE SALES TERMS YOU MAY NOT PROCEED TO PURCHASE ANY PRODUCTS LISTED ON THE PANDORA WEBSITE.\n2. Your Account\nIf you create an account (personal user identification or username) on the Pandora website, you are responsible for maintaining the confidentiality of your username and clave and for restricting access to your clave.\nBy creating an account, you agree to accept responsibility for all activities and purchases that occur under your account, unless those activities or purchases are carried out by an unauthorised third party who has gained access to your account other than by reason of your negligence or carelessness.\nPandora makes no guarantee that the Pandora website or your account will be error free or secure from bugs or viruses. You are responsible for your own information technology and security and you should use your own virus protection software. Please tell us immediately if your account or payment details have been lost or stolen. If we become concerned about the security of your account we may block access and usage.\n3. Placing an Order\nTo be able to buy Products you must:\n• read and accept these Sales Terms;\n• provide your name and address, phone number, email address, payment details and other required information;\n• provide an eligible delivery address in Great Britain or Northern Ireland (see below for further details);\n• be legally capable of entering into binding contracts; and\n• be the owner or authorised holder of a valid debit/credit card, gift card or Apple Pay account (see\nsection 6 (Payment)\nfor details of the payment methods accepted on the Pandora website).\nYou may place an order by clicking on the "ADD TO BAG" button and proceeding to the checkout page.\nAt the checkout page you will be provided with an opportunity to review your order(s), read and accept these Sales Terms, check the total price of your order(s) and the information you have provided, and correct any input errors before confirming your order(s).\nYou must check and ensure that your delivery address is correct before placing your order as it may not be possible to change your address before the order is dispatched. Please see\nsection 4 (Order Processing)\nbelow which deals with this in more detail.\nThe Pandora website currently ships to all addresses within Great Britain and Northern Ireland excluding hotels, hostels, prisons, P.O. Boxes and BFPO (British Forces Post Office) addresses. We do not currently deliver to Channel Islands.\n4. Order Processing and Contract Formation\nOnce you have submitted your order(s) using the Pandora website, Pandora will send you an email acknowledging receipt of your order(s) ("Order Receipt Confirmation") and setting out details of the ordered Product(s).\nPandora may not always be able to accept your order and Pandora may choose not to accept your order in its sole discretion. For example (without limitation) there may be pricing errors, insufficient stock, unusual orders or orders which Pandora suspects are not placed in good faith. There may also be unexpected events outside of our control meaning we may be unable to fulfil your order.\nIf your order is not accepted Pandora will inform you via email, and if any payment has already been taken you will be reimbursed in full for any sums paid to Pandora for that order.\nAfter receiving your order(s), Pandora will check that the relevant Product(s) is/are in stock. A contract for the sale of Products between you and Pandora, which will be governed by these Sale Terms, will only be formed when Pandora emails you confirming that all or part the Product(s) is/are still available and has/have been dispatched ("Shipment Confirmation").\nThe contract is formed upon Pandora sending the Shipment Confirmation to you at the email address provided with your order. Please contact Pandora\'s Customer Care Team if you have not received your Shipment Confirmation within 72 hours of placing your order.\nA contract for the sale of Products will relate only to those Products whose dispatch has been confirmed in the Shipment Confirmation. Pandora will not be obliged to supply any other Products which may have been part of your order until the dispatch of such Products has been confirmed in a separate Shipment Confirmation.\nIn the Order Receipt Confirmation you will receive a copy of these Sales Terms, however you are encouraged to download, save and/or print a copy of your Order Receipt Confirmation and these Sales Terms for your records.\nIf you would like to amend or cancel your order before you have received your Shipment Confirmation, you must contact the Pandora Customer Care Team as soon as possible using one of the methods set out in\nsection 17 (Contact Details)\nor as otherwise set out in the Order Receipt Confirmation. It may not be possible to amend or cancel your order, or to prevent dispatch of the Products, due to the speed with which orders are processed. If Pandora is unable to amend or cancel your order and the Products are dispatched to you, you have the option of either refusing to accept the delivery, in which case the Products will be returned to Pandora, or accepting the delivery and then returning the Products to Pandora in accordance with\nsection 12 (Returns and Refunds)\n.\nOnce you have received your Shipment Confirmation, it will be too late to prevent the Products from being dispatched to you. If you would like to cancel your order after this point, please see\nsection 12 (Returns and Refunds)\nbelow.\n5. Prices, Shipping and Handling, Charges and Taxes\nThe price charged for a Product will be the price in effect at the time your order is placed and will be set out in the Order Receipt Confirmation and in your Shipment Confirmation. All prices are payable in Great British Pound Sterling (GBP).\nIn the event of a pricing error, whether on the Pandora website or in an Order Receipt Confirmation or otherwise, you accept that Pandora has the right to correct such error and charge the correct price or cancel the order. If the actual price is more than the amount shown on your Order Receipt Confirmation, Pandora will notify you of the correct price and give you the option of proceeding with the order or cancelling it. If we are unable to contact you or if we do not receive a response from you, your order will be cancelled and any amounts paid by you to Pandora for the Products will be reimbursed. If you do not want to proceed with your order your sole remedy in the event of such a pricing error is to cancel your order.\nYou acknowledge that even after Pandora has sent you the Shipment Confirmation, if the pricing error is obvious and could reasonably have been recognised by you as an error, then Pandora will be under no obligation to provide the relevant Product(s) to you at the incorrect price.\nPrices for the Products include VAT and/or other applicable taxes but do not include charges for shipping and handling. Separate charges for shipping and handling (and their related VAT or other taxes) will be clearly displayed on the screen prior to the completion of your order and will be confirmed in your Order Receipt Confirmation.\nPrices for Products and for shipping and handling are liable to change at any time, but changes will not normally affect orders in respect of which we have already sent you a Shipment Confirmation (save in the case of a manifest pricing error as described above).\n6. Payment\nPayment shall be made by one of the methods you have selected during the checkout process (being VISA, MasterCard, Maestro, PayPal, Apple Pay or Pandora Gift Card).\nApproved credit and debit card types are listed at the Pandora website checkout. In the event Pandora does not receive the appropriate authorisation from your card or payment service provider, Pandora reserves the right to reject your order.\nPayment for your order, including all applicable taxes, shipping and other charges, will be settled from the applicable card or payment service once Pandora has sent you the Shipment Confirmation.\n7. Delivery\nYou should allow up to 10 days (excluding Sunday and bank holidays) for standard delivery and up to 3 business days for express delivery (orders must be placed by 8pm) for in stock items (check\nhere\nfor current delivery periods). Delivery shall in any event take place within 30 days after Pandora has sent you the Shipment Confirmation. Pandora reserves the right to substitute another carrier of equal or lesser cost to deliver your order. The delivery options described above are the only shipping options available for sales through the Pandora website. Pandora is unable to accept any special delivery requests, which are outside the scope of these two options.\nIf Pandora cannot deliver your order within the period specified in your Shipment Confirmation, Pandora will notify you as soon as possible. You may choose to cancel your order, in which case Pandora will provide you with a full refund. If you choose not to cancel your order, Pandora will endeavour to deliver your order within a reasonable period of time.\nIf you unreasonably defer delivery or delay the receipt of delivery after Pandora has notified you that the carrier has tried to deliver the ordered items to you, or if you have provided Pandora with an incorrect delivery address which results in an unsuccessful delivery, the delivery package will be returned to Pandora, your order will be cancelled and a full refund will be issued to you.\n8. Events Outside the Control of Pandora\nPandora shall not be held responsible for delay or failure to perform, if the delay or failure is caused by any act, event, non-happening, omission, accident or circumstance beyond its reasonable control, including, but not limited to:\n• civil commotion, riot, invasion, terrorist attack or threat of terrorist attack, war (whether declared or not) or threat or preparation for war;\n• strikes or lock-outs (or other industrial action);\n• national or local states of emergency;\n• interruption or failure of communication or transportation facilities;\n• power or utility outages;\n• non-performance by suppliers or subcontractors (other than by companies in the same group of companies as Pandora A/S);\n• earthquake, fire, explosion, storm, flood, subsidence, epidemic or other natural disaster;\n• impossibility of the use of railways, shipping, aircraft, motor transport or other means of public or private transport;\n• pandemic, epidemic or public health crisis\n• the acts, decrees, legislation, regulations or restrictions of any government.\nIn such circumstances, Pandora will notify you and take reasonable steps to minimise any delay. The time for Pandora\'s performance of its obligations is deemed to be extended for the period that any of the above continues. If Pandora is unable to fulfil your order because of any circumstances outside of its control, or if there is a substantial delay to our delivering your order, you may cancel your order and obtain a full refund.\n9. Product Availability\nPandora uses reasonable efforts to ensure the availability of Products you order from the Pandora website. However, Products are made available to purchase on a "first in-first out" basis and are only available while stocks last.\n10. Compatibility, Product Information\nPlease take care when placing your order to ensure that the Products you purchase are compatible for the intended use. Please use the Pandora website as your final point of reference when checking compatibility. In the event of a difference between the Pandora website content and any other website (or any other source of information) the compatibility of Products as shown on the Pandora website at the time of purchase will be seen as taking precedence.\nImages on the Pandora website may vary depending on your device\'s display quality. Whilst every effort is made to reproduce images of the Products in the most accurate way possible, these images are for illustrative purposes only and your Products may vary from those images for example, in colour, size, pattern and texture. The packaging of Products may also vary from that shown in images on the Pandora website.\nIf the Products you have received do not correspond to the ones you have ordered, or if your delivery is incomplete or damaged in transportation, please contact the Pandora\nCustomer Care Team\nfor assistance without delay, either via email or by phone.\n11. Title to Products\nOwnership of the Products will pass to you only when Pandora has received full payment in respect of your order, including all taxes, shipping and other charges. The Products will be at your risk from the time of delivery.\n12. Return and Refund\nCancellations under the Consumer Contracts Regulations\nUnder the Consumer Contracts (Information, Cancellation and Additional Charges) Regulations 2013 (“Consumer Contracts Regulations”), you have a legal right to cancel your purchase during the cancellation period. The cancellation period begins on the date that the contract is formed (when Pandora sends you the Shipment Confirmation) and continues until the end of 14 days starting on the day after you receive the Product. This cancellation right applies to both sale and non-sale Products but does not apply to Excluded Products as set out below. If the Product has been dispatched to you before you cancelled the purchase, you must return the Product to Pandora within 14 days starting on the day after you cancel the order. If you wish to cancel your purchase under the Consumer Contracts\n                                              Regulations, you can take advantage of the Pandora website’s free Returns Process. You are not obliged to follow the free\nReturns Process\n, but if you do not do so you will be solely responsible for the cost of shipping any Products back to Pandora.\nIf you do not wish to use the free Returns Process, you can cancel your purchase by notifying the Customer Care Team either by using our\nContact Us\nform or by telephone to the Freephone number 0808 2345 431 (check\nopening hours here\nor by ordinary mail to Pandora Jewellery UK Limited at Ecommerce Department, 33 George Street, London, W1U 3BH. You may also use this\nWithdrawal Form\nto cancel your purchase (and, if you do so, Pandora will confirm receipt by email). Alternatively, you can simply return your Product to Pandora at PFS Conexus, Unit 7 Mountpark Southampton Wide Lane, Southampton, SO18 2NQ within the 14 day period starting on the day after delivery, in which case your return will be treated as your notice of cancellation. When you return your Products to Pandora, please include your name, email address, order number and order date to help Pandora identify your order quickly.\nIf you cancel your purchase under the Consumer Contracts Regulations, Pandora will refund to you the full price you paid for the Product (including all VAT and/or other taxes) together with delivery costs you have paid, although, as permitted by law, the maximum delivery costs refunded will be the cost of the standard delivery service.\nOnce Pandora has received and validated your returned Product, Pandora will send you an email to confirm the validation. Any refund due to you will be made as soon as possible and in any event within the period of 14 days after the day on which Pandora receives the Product back from you or, if earlier, the day on which you provide Pandora with evidence that you have sent the Product back to Pandora. If the Product was not dispatched to you, you will be refunded within the period of 14 days after you inform Pandora of your decision to cancel the purchase. Pandora will refund you by the same method you used to make payment.\nPandora is entitled by law to reduce your refund to reflect any reduction in the value of the Product, if this has been caused by your handling it in a way which would not be permitted in a shop. If Pandora refunds the price paid before it has an opportunity to inspect the Product and later discovers you have handled it in an unacceptable way or damaged it, you must pay Pandora an appropriate amount.\nReturns under the Pandora website enhanced cancellation policy\nIn addition to your rights under the Consumer Contracts Regulations, if for any reason you are not completely satisfied with your purchase from the Pandora website, you can return non-sale Products purchased from the Pandora website within 30 days of receipt of the Products for a refund, but to do so you must follow the free\nReturns Process\n.\nOnce Pandora has received and validated any returned Products, Pandora will send you an email to confirm the validation. Your refund will be processed within 14 (fourteen) days of Pandora’s email to you confirming the receipt and validation of your returned Product.\nAny Product returned under the Pandora website enhanced cancellation policy must be in a new, unworn and resaleable condition and Pandora reserves the right to refuse a refund where the Product is returned damaged or showing signs of use or wear.\nPlease note that these enhanced return rights do not apply to sale Products or to any Excluded Products. This does not affect your rights in law as a consumer including your right to return products under the Consumer Contracts Regulations.\nExcluded Products\nThe Pandora website does not accept returns or exchanges of: (i) Pandora Gift Cards or eGift Cards; (ii) products purchased from any Pandora store, Pandora concession store, Pandora outlet store, from stores which do not exclusively sell Pandora products, or from any overseas Pandora store; (iii) any bespoke or customised products.\nReturning Products that are faulty or not as described\nPlease see\nsection 14 (Consumer Warranty (Guarantee))\nbelow for more information about your legal rights in relation to Products that are faulty or not as described.\nIf you need to return any Product because you believe it is faulty or not as described, but the deadline for returning the Product under the\nConsumer Contracts Regulations\nand the\nPandora website enhanced cancellation policy\n(if applicable) have expired, please contact the Customer Care Team for further assistance, either by using our\nContact Us\nform, by telephone to the Freephone number 0808 2345 431) check\nopening hours here\nor by ordinary mail to Pandora at PFS Conexus, Unit 7 Mountpark Southampton Wide Lane, Southampton, SO18 2NQ.\nPandora will refund in full the price of Products (including sale Products) that are faulty or not as described, together with all applicable delivery charges (including any premium paid for non-standard delivery), and any reasonable costs you incur in returning the item.\nExchanges through the Pandora website\nProducts purchased from the Pandora website (including sale items) can also be exchanged in a Pandora store and certain Pandora stockists within 30 days from purchase, but only if you wish to exchange it for the same product in a different size or for a different product of exactly the same value. Please be aware that Pandora stockists have their own exchange policy which may differ to Pandora\'s, please contact the stockist directly for details.\nDue to a higher than usual volume of enquiries, we have temporarily suspended Product exchanges through the Pandora website. This means that you cannot exchange Products through the Pandora website. However, you may return Products to the Pandora website using the Consumer Contracts Regulations process set out above.\nExceptions\nIf you do not return a Product (or, if a Product sent using the free\nReturns Process\nfails to arrive and you are unable to provide proof of postage), Pandora reserves the right to refuse a refund.\n13. Promotions\nIf you participate in a Pandora promotion additional terms apply. For more information please read our\nPromotion FAQs\n.\n14. Consumer Warranty (Guarantee)\nTHIS GUARANTEE GIVES YOU SPECIFIC RIGHTS. IT IS OFFERED AS AN ADDITIONAL BENEFIT TO YOUR RIGHTS UNDER LAW. YOU ALSO HAVE OTHER RIGHTS UNDER LAW WHICH MAY VARY FROM COUNTRY TO COUNTRY.\nYou can find out about your legal rights and remedies by contacting your local consumer advice service. In particular, the Pandora website has a legal duty to supply products that conform to the contract made with you. Nothing in this Guarantee will replace or lessen any of your legal rights or remedies.\nProducts sold by the Pandora website include a guarantee for Products which turn out to be faulty or not as described during the periods set out below for various Product types:\n• 2 years from the original date of purchase for silver jewellery\n• 2 years from the original date of purchase for gold jewellery\n• 2 years from the original date of purchase for Pandora Rose and Pandora Shine jewellery\n• 1 year from the original date of purchase for parts of Products made of wood, leather, glass and string items.\nThis means you can request a repair or replacement free of charge if the Products turn out to be faulty or not as described during this period. Pandora will repair your Product or replace it with a Product of identical or similar characteristics, except as set out below. Pandora may at its own discretion replace your Product under this Guarantee whether or not a repair of your original Product is possible. If Pandora chooses instead to repair your Product Pandora may do so with new or refurbished components.\nIf the Product cannot be repaired or replaced within a reasonable time or without inconvenience, you may request a refund or price reduction. Refunds, if authorised, will usually be made using the same payment method originally used by you to pay for your purchase.\nPlease retain your Order Receipt Confirmation or pass it on to the gift recipient as the original proof of purchase is the warranty under which this Guarantee operates. If the Order Receipt Confirmation is not available, a credit card or bank statement showing the relevant transaction will suffice. This does not affect your rights in law as a consumer.\nThis Guarantee does not cover normal wear and tear or damage caused by accidents, mishandling, improper use (such as knocks, dents, crushing etc.).\nRepairs (or attempted repairs) to Products other than by Pandora will void this Guarantee whether or not damage has been caused by such repair or attempted repair.\nIf your Product is repaired or replaced by Pandora under this Guarantee, the new item will benefit from the remainder of the term of this Guarantee (calculated from the date of the original purchase of the Product). The period of this Guarantee shall not be extended whether or not your Product is repaired or replaced.\nWearing counterfeit charms, charms from incompatible Pandora collections or charms by other brands on your Pandora bracelet could damage the unique threading system and if such damage is caused, it will not be covered by this Guarantee.\n15. Liability\nPandora warrants to you that any Product purchased from the Pandora website is of satisfactory quality and fit for all the purposes for which products of the same kind are commonly supplied.\nPandora is only liable for foreseeable loss and damage. Pandora is not responsible for any loss or damage that is not foreseeable. HOWEVER NOTHING IN THESE SALES TERMS AFFECTS YOUR LEGAL RIGHTS AS A CONSUMER. TO THE FULL EXTENT PERMITTED BY LAW, THE TOTAL LIABILITY OF PANDORA AND PANDORA A/S SHALL BE LIMITED TO THE AMOUNT ACTUALLY PAID FOR THE PURCHASE OF THE APPLICABLE PRODUCTS PLUS DELIVERY FROM THE PANDORA WEBSITE.\nIn particular, nothing in these Sales Terms excludes or limits in any way the liability of Pandora:\n• for death or personal injury caused by Pandora\'s negligence;\n• for fraud or fraudulent misrepresentation;\n• under section 2(3) of the UK\'s Consumer Protection Act 1987 or for any breach of your consumer rights relating to products; or\n• for any matter for which it would be illegal for Pandora to limit or exclude, or attempt to limit or exclude, liability.\n16. Severability\nIf any provision of these Sales Terms shall be deemed unlawful, void or for any reason unenforceable, then that provision shall be deemed severable (deleted) from these Sales Terms and this will not affect the validity and enforceability of any remaining provisions.\n17. Contact Details\nPandora\'s\nCustomer Care Team\nwill assist you with any Pandora website order related questions or complaints. You can contact the Pandora Customer Care Team in the following ways:\n• via the\nContact Us\nform on the Pandora website;\n• by FREEPHONE 0808 2345 431 check\nopening hours here\nor\n• in writing to Pandora Jewellery UK Limited at Ecommerce Department, 33 George Street, London, W1U 3BH.\n*Please note, calls are free from any UK landline. UK mobile network charges might apply. Please check with your network provider.\nPandora will endeavour to resolve any complaint you may have. If Pandora is unable to resolve a dispute, either party is entitled to seek further recourse through the\nODR Platform\n. Operated by the European Commission, the ODR Platform is an online platform providing businesses and customers in the European Union with a forum for resolving online sales disputes without the need to go to court. The dispute resolution services available on the Platform are provided free of charge, although neither party is under any obligation to participate.\n18. Transferring rights and obligations to another person or organisation\nThese Sales Terms are binding on you and Pandora. It will also be binding on any person or organisation to which Pandora might transfer its rights and obligations to.\nYou may not transfer or otherwise dispose of your contract with Pandora, or any of your rights or obligations arising under it, without Pandora\'s prior written consent.\nPandora may transfer or otherwise dispose of a contract with you, or transfer any of Pandora\'s rights or obligations arising under it, at any time. If you have an outstanding order with Pandora at the time of any such disposal or transfer, you will be contacted if Pandora and you may cancel your outstanding order if you do not want to proceed.\n19. Third Party Rights\nOther than Pandora A/S, any person who is not a party to the contract between you and Pandora is not entitled to enforce any of its terms.\n20. Intellectual Property Rights\nThe content on the Pandora website including all jewellery design, copyright, trade marks and other intellectual property rights it contains, including the name Pandora is the sole property of Pandora or its licensors.\n21. Data Protection\nPandora believes strongly in protecting user privacy and personal data. Any personal data collected and processed by Pandora is governed by Pandora\'s\nPrivacy Policy\n. Please read Pandora\'s\nPrivacy Policy\nfor further information on how Pandora processes personal data collected from users of the Pandora website.\n22. Waiver\nIf Pandora fails, at any time, to insist that you comply with your obligations under these Sales Terms, or if Pandora does not exercise any of its rights under these Sales Terms, this does not constitute a waiver of such rights and does not mean that you are free to ignore your obligations. Pandora can still require you to comply at a later date.\n23. Entire Agreement\nThese Sales Terms and any document that we have referred to within them (including via hyperlink) represent the entire agreement between Pandora and you and takes precedence over any other previous agreement or representation made to you whether oral or in writing.\nYou acknowledge that, in entering into a contract with Pandora, you have not relied on any representation or promise given by Pandora or anyone else except as set out in these Sales Terms.\n24. Governing Law and Jurisdiction\nThese Sales Terms and your purchase of Products from Pandora through the Pandora website shall be governed by and construed in accordance with the laws of England.\nAny dispute or claim arising out of or in connection with these Sales Terms or your purchase of Products from Pandora (including non-contractual disputes or claims) shall be subject to the non-exclusive jurisdiction of the English courts. If you are a resident of Northern Ireland, Scotland or Wales you may also bring proceedings in your local courts\nLast updated\n15 June 2020"""
    generated_content = """PANDORA\nView in\nbrowser\n.\nNEW\nCHARMS\nBRACELETS\nRINGS\nNECKLACES\nEARRINGS\nTHANK YOU FOR YOUR ORDER\nWe have received your order and are preparing it for despatch.\nExpress delivery orders should reach you in 1-2 working days. Standard delivery orders should reach you between 2-4 working days. Pandora\'s response to COVID-19:\nRead more\nORDER SUMMARY\nMy Spacer Micro Charm\nSize: One size\nMetal: Sterling Silver\nQuantity: 1\nUnit price:\n£10.00\nORDER SUB-TOTAL:\n£10.00\nSHIPPING:\n£2.99\nDISCOUNT:\n£0.00\nTOTAL INCL VAT:\n£12.99\nVAT AMOUNT:\n£2.17\nPAYMENT METHOD:\nPAYPAL\nYOUR ADDRESSES\nSHIPPING INFORMATION\nCitadel Investment Group Europe Ltd\n   \nBILLING INFORMATION\nCitadel Investment Group Europe Ltd\n   \nHAVE A QUESTION ABOUT YOUR ORDER?\nLet us help you answer it. Browse our FAQs, send us an email, or speak to our Customer Care team.\n                                                                      We are not able to process exchanges at this time.\nMore information here to return your order.\nHELP IS HERE\nJOIN PANDORA CLUB\nAs a Pandora club member, you enjoy easy checkout, special treats and much more.\nBECOME A MEMBER\nWith love,\nPandora\nMY ACCOUNT\nPANDORA CLUB\nFAQS\nCONTACT US\nSTORE LOCATOR\nMY ACCOUNT\nPANDORA CLUB\nFAQS\nCONTACT\nSTORE LOCATOR\nSTANDARD DELIVERY\n£2.99\nFREE AND EASY\nUK RETURNS\nPRIVACY POLICY\nTERMS & CONDITIONS\nCOOKIE POLICY\nPRIVACY POLICY\nTERMS & CONDITIONS\nCOOKIE POLICY\nPlease note: It is not possible to return an online order in store. Exchange only. For full details of how to return your item for a refund, please visit our\nreturns\npage.\nPlease do not reply to this email. If you would like to contact us please\nclick here\n.\nTerms & Conditions apply.\n©Pandora. All rights reserved.\nTERMS AND CONDITIONS\nIt is important that you read these terms and conditions ("Sales Terms") carefully before ordering any Products from the Pandora website. Your purchase of Products via the Pandora website will be governed by the Sale Terms. By ordering any Products, you agree to be bound by the Sale Terms.\n1. About These Sales Terms\nThese Sales Terms are applicable to all orders and purchases made via the Pandora website. Pandora may revise these Sales Terms from time to time. Any changes to the Sales Terms will apply on or after the date that the revised Sales Terms are published. You are advised to check this page from time to time to take notice of any changes, as they are binding on you.\nPandora offers products for sale via the Pandora website to consumer customers. Pandora reserves the right to not fulfil orders placed by non-consumers or by individuals who fail to comply with these terms.\nWith regard to purchases made by you via the Pandora website, Pandora will usually communicate with you electronically by email, using the email address provided to Pandora when you place an order or otherwise communicate with Pandora. You may still contact us using one of the methods referred to in\nsection 17 (Contact Details)\nbelow or by using the contact details on the Pandora website.\nIn using the Pandora website you agree to comply with\nPandora\'s Acceptable Use Policy\n.\nBEFORE PLACING AN ORDER FOR PRODUCTS YOU MUST READ AND AGREE TO BE BOUND BY THESE SALES TERMS.\nYOUR ATTENTION IS DRAWN IN PARTICULAR TO THE PROVISIONS OF\nSECTION 15 (LIABILITY)\n.\n                                              IF YOU FIND YOURSELF UNABLE TO AGREE TO THESE SALES TERMS YOU MAY NOT PROCEED TO PURCHASE ANY PRODUCTS LISTED ON THE PANDORA WEBSITE.\n2. Your Account\nIf you create an account (personal user identification or username) on the Pandora website, you are responsible for maintaining the confidentiality of your username and clave and for restricting access to your clave.\nBy creating an account, you agree to accept responsibility for all activities and purchases that occur under your account, unless those activities or purchases are carried out by an unauthorised third party who has gained access to your account other than by reason of your negligence or carelessness.\nPandora makes no guarantee that the Pandora website or your account will be error free or secure from bugs or viruses. You are responsible for your own information technology and security and you should use your own virus protection software. Please tell us immediately if your account or payment details have been lost or stolen. If we become concerned about the security of your account we may block access and usage.\n3. Placing an Order\nTo be able to buy Products you must:\n• read and accept these Sales Terms;\n• provide your name and address, phone number, email address, payment details and other required information;\n• provide an eligible delivery address in Great Britain or Northern Ireland (see below for further details);\n• be legally capable of entering into binding contracts; and\n• be the owner or authorised holder of a valid debit/credit card, gift card or Apple Pay account (see\nsection 6 (Payment)\nfor details of the payment methods accepted on the Pandora website).\nYou may place an order by clicking on the "ADD TO BAG" button and proceeding to the checkout page.\nAt the checkout page you will be provided with an opportunity to review your order(s), read and accept these Sales Terms, check the total price of your order(s) and the information you have provided, and correct any input errors before confirming your order(s).\nYou must check and ensure that your delivery address is correct before placing your order as it may not be possible to change your address before the order is dispatched. Please see\nsection 4 (Order Processing)\nbelow which deals with this in more detail.\nThe Pandora website currently ships to all addresses within Great Britain and Northern Ireland excluding hotels, hostels, prisons, P.O. Boxes and BFPO (British Forces Post Office) addresses. We do not currently deliver to Channel Islands.\n4. Order Processing and Contract Formation\nOnce you have submitted your order(s) using the Pandora website, Pandora will send you an email acknowledging receipt of your order(s) ("Order Receipt Confirmation") and setting out details of the ordered Product(s).\nPandora may not always be able to accept your order and Pandora may choose not to accept your order in its sole discretion. For example (without limitation) there may be pricing errors, insufficient stock, unusual orders or orders which Pandora suspects are not placed in good faith. There may also be unexpected events outside of our control meaning we may be unable to fulfil your order.\nIf your order is not accepted Pandora will inform you via email, and if any payment has already been taken you will be reimbursed in full for any sums paid to Pandora for that order.\nAfter receiving your order(s), Pandora will check that the relevant Product(s) is/are in stock. A contract for the sale of Products between you and Pandora, which will be governed by these Sale Terms, will only be formed when Pandora emails you confirming that all or part the Product(s) is/are still available and has/have been dispatched ("Shipment Confirmation").\nThe contract is formed upon Pandora sending the Shipment Confirmation to you at the email address provided with your order. Please contact Pandora\'s Customer Care Team if you have not received your Shipment Confirmation within 72 hours of placing your order.\nA contract for the sale of Products will relate only to those Products whose dispatch has been confirmed in the Shipment Confirmation. Pandora will not be obliged to supply any other Products which may have been part of your order until the dispatch of such Products has been confirmed in a separate Shipment Confirmation.\nIn the Order Receipt Confirmation you will receive a copy of these Sales Terms, however you are encouraged to download, save and/or print a copy of your Order Receipt Confirmation and these Sales Terms for your records.\nIf you would like to amend or cancel your order before you have received your Shipment Confirmation, you must contact the Pandora Customer Care Team as soon as possible using one of the methods set out in\nsection 17 (Contact Details)\nor as otherwise set out in the Order Receipt Confirmation. It may not be possible to amend or cancel your order, or to prevent dispatch of the Products, due to the speed with which orders are processed. If Pandora is unable to amend or cancel your order and the Products are dispatched to you, you have the option of either refusing to accept the delivery, in which case the Products will be returned to Pandora, or accepting the delivery and then returning the Products to Pandora in accordance with\nsection 12 (Returns and Refunds)\n.\nOnce you have received your Shipment Confirmation, it will be too late to prevent the Products from being dispatched to you. If you would like to cancel your order after this point, please see\nsection 12 (Returns and Refunds)\nbelow.\n5. Prices, Shipping and Handling, Charges and Taxes\nThe price charged for a Product will be the price in effect at the time your order is placed and will be set out in the Order Receipt Confirmation and in your Shipment Confirmation. All prices are payable in Great British Pound Sterling (GBP).\nIn the event of a pricing error, whether on the Pandora website or in an Order Receipt Confirmation or otherwise, you accept that Pandora has the right to correct such error and charge the correct price or cancel the order. If the actual price is more than the amount shown on your Order Receipt Confirmation, Pandora will notify you of the correct price and give you the option of proceeding with the order or cancelling it. If we are unable to contact you or if we do not receive a response from you, your order will be cancelled and any amounts paid by you to Pandora for the Products will be reimbursed. If you do not want to proceed with your order your sole remedy in the event of such a pricing error is to cancel your order.\nYou acknowledge that even after Pandora has sent you the Shipment Confirmation, if the pricing error is obvious and could reasonably have been recognised by you as an error, then Pandora will be under no obligation to provide the relevant Product(s) to you at the incorrect price.\nPrices for the Products include VAT and/or other applicable taxes but do not include charges for shipping and handling. Separate charges for shipping and handling (and their related VAT or other taxes) will be clearly displayed on the screen prior to the completion of your order and will be confirmed in your Order Receipt Confirmation.\nPrices for Products and for shipping and handling are liable to change at any time, but changes will not normally affect orders in respect of which we have already sent you a Shipment Confirmation (save in the case of a manifest pricing error as described above).\n6. Payment\nPayment shall be made by one of the methods you have selected during the checkout process (being VISA, MasterCard, Maestro, PayPal, Apple Pay or Pandora Gift Card).\nApproved credit and debit card types are listed at the Pandora website checkout. In the event Pandora does not receive the appropriate authorisation from your card or payment service provider, Pandora reserves the right to reject your order.\nPayment for your order, including all applicable taxes, shipping and other charges, will be settled from the applicable card or payment service once Pandora has sent you the Shipment Confirmation.\n7. Delivery\nYou should allow up to 10 days (excluding Sunday and bank holidays) for standard delivery and up to 3 business days for express delivery (orders must be placed by 8pm) for in stock items (check\nhere\nfor current delivery periods). Delivery shall in any event take place within 30 days after Pandora has sent you the Shipment Confirmation. Pandora reserves the right to substitute another carrier of equal or lesser cost to deliver your order. The delivery options described above are the only shipping options available for sales through the Pandora website. Pandora is unable to accept any special delivery requests, which are outside the scope of these two options.\nIf Pandora cannot deliver your order within the period specified in your Shipment Confirmation, Pandora will notify you as soon as possible. You may choose to cancel your order, in which case Pandora will provide you with a full refund. If you choose not to cancel your order, Pandora will endeavour to deliver your order within a reasonable period of time.\nIf you unreasonably defer delivery or delay the receipt of delivery after Pandora has notified you that the carrier has tried to deliver the ordered items to you, or if you have provided Pandora with an incorrect delivery address which results in an unsuccessful delivery, the delivery package will be returned to Pandora, your order will be cancelled and a full refund will be issued to you.\n8. Events Outside the Control of Pandora\nPandora shall not be held responsible for delay or failure to perform, if the delay or failure is caused by any act, event, non-happening, omission, accident or circumstance beyond its reasonable control, including, but not limited to:\n• civil commotion, riot, invasion, terrorist attack or threat of terrorist attack, war (whether declared or not) or threat or preparation for war;\n• strikes or lock-outs (or other industrial action);\n• national or local states of emergency;\n• interruption or failure of communication or transportation facilities;\n• power or utility outages;\n• non-performance by suppliers or subcontractors (other than by companies in the same group of companies as Pandora A/S);\n• earthquake, fire, explosion, storm, flood, subsidence, epidemic or other natural disaster;\n• impossibility of the use of railways, shipping, aircraft, motor transport or other means of public or private transport;\n• pandemic, epidemic or public health crisis\n• the acts, decrees, legislation, regulations or restrictions of any government.\nIn such circumstances, Pandora will notify you and take reasonable steps to minimise any delay. The time for Pandora\'s performance of its obligations is deemed to be extended for the period that any of the above continues. If Pandora is unable to fulfil your order because of any circumstances outside of its control, or if there is a substantial delay to our delivering your order, you may cancel your order and obtain a full refund.\n9. Product Availability\nPandora uses reasonable efforts to ensure the availability of Products you order from the Pandora website. However, Products are made available to purchase on a "first in-first out" basis and are only available while stocks last.\n10. Compatibility, Product Information\nPlease take care when placing your order to ensure that the Products you purchase are compatible for the intended use. Please use the Pandora website as your final point of reference when checking compatibility. In the event of a difference between the Pandora website content and any other website (or any other source of information) the compatibility of Products as shown on the Pandora website at the time of purchase will be seen as taking precedence.\nImages on the Pandora website may vary depending on your device\'s display quality. Whilst every effort is made to reproduce images of the Products in the most accurate way possible, these images are for illustrative purposes only and your Products may vary from those images for example, in colour, size, pattern and texture. The packaging of Products may also vary from that shown in images on the Pandora website.\nIf the Products you have received do not correspond to the ones you have ordered, or if your delivery is incomplete or damaged in transportation, please contact the Pandora\nCustomer Care Team\nfor assistance without delay, either via email or by phone.\n11. Title to Products\nOwnership of the Products will pass to you only when Pandora has received full payment in respect of your order, including all taxes, shipping and other charges. The Products will be at your risk from the time of delivery.\n12. Return and Refund\nCancellations under the Consumer Contracts Regulations\nUnder the Consumer Contracts (Information, Cancellation and Additional Charges) Regulations 2013 (“Consumer Contracts Regulations”), you have a legal right to cancel your purchase during the cancellation period. The cancellation period begins on the date that the contract is formed (when Pandora sends you the Shipment Confirmation) and continues until the end of 14 days starting on the day after you receive the Product. This cancellation right applies to both sale and non-sale Products but does not apply to Excluded Products as set out below. If the Product has been dispatched to you before you cancelled the purchase, you must return the Product to Pandora within 14 days starting on the day after you cancel the order. If you wish to cancel your purchase under the Consumer Contracts\n                                              Regulations, you can take advantage of the Pandora website’s free Returns Process. You are not obliged to follow the free\nReturns Process\n, but if you do not do so you will be solely responsible for the cost of shipping any Products back to Pandora.\nIf you do not wish to use the free Returns Process, you can cancel your purchase by notifying the Customer Care Team either by using our\nContact Us\nform or by telephone to the Freephone number 0808 2345 431 (check\nopening hours here\nWithdrawal Form\nto cancel your purchase (and, if you do so, Pandora will confirm receipt by email). Alternatively, you can simply return your Product to Pandora at PFS Conexus, Unit 7 Mountpark Southampton Wide Lane, Southampton, SO18 2NQ within the 14 day period starting on the day after delivery, in which case your return will be treated as your notice of cancellation. When you return your Products to Pandora, please include your name, email address, order number and order date to help Pandora identify your order quickly.\nIf you cancel your purchase under the Consumer Contracts Regulations, Pandora will refund to you the full price you paid for the Product (including all VAT and/or other taxes) together with delivery costs you have paid, although, as permitted by law, the maximum delivery costs refunded will be the cost of the standard delivery service.\nOnce Pandora has received and validated your returned Product, Pandora will send you an email to confirm the validation. Any refund due to you will be made as soon as possible and in any event within the period of 14 days after the day on which Pandora receives the Product back from you or, if earlier, the day on which you provide Pandora with evidence that you have sent the Product back to Pandora. If the Product was not dispatched to you, you will be refunded within the period of 14 days after you inform Pandora of your decision to cancel the purchase. Pandora will refund you by the same method you used to make payment.\nPandora is entitled by law to reduce your refund to reflect any reduction in the value of the Product, if this has been caused by your handling it in a way which would not be permitted in a shop. If Pandora refunds the price paid before it has an opportunity to inspect the Product and later discovers you have handled it in an unacceptable way or damaged it, you must pay Pandora an appropriate amount.\nReturns under the Pandora website enhanced cancellation policy\nIn addition to your rights under the Consumer Contracts Regulations, if for any reason you are not completely satisfied with your purchase from the Pandora website, you can return non-sale Products purchased from the Pandora website within 30 days of receipt of the Products for a refund, but to do so you must follow the free\nReturns Process\n.\nOnce Pandora has received and validated any returned Products, Pandora will send you an email to confirm the validation. Your refund will be processed within 14 (fourteen) days of Pandora’s email to you confirming the receipt and validation of your returned Product.\nAny Product returned under the Pandora website enhanced cancellation policy must be in a new, unworn and resaleable condition and Pandora reserves the right to refuse a refund where the Product is returned damaged or showing signs of use or wear.\nPlease note that these enhanced return rights do not apply to sale Products or to any Excluded Products. This does not affect your rights in law as a consumer including your right to return products under the Consumer Contracts Regulations.\nExcluded Products\nThe Pandora website does not accept returns or exchanges of: (i) Pandora Gift Cards or eGift Cards; (ii) products purchased from any Pandora store, Pandora concession store, Pandora outlet store, from stores which do not exclusively sell Pandora products, or from any overseas Pandora store; (iii) any bespoke or customised products.\nReturning Products that are faulty or not as described\nPlease see\nsection 14 (Consumer Warranty (Guarantee))\nbelow for more information about your legal rights in relation to Products that are faulty or not as described.\nIf you need to return any Product because you believe it is faulty or not as described, but the deadline for returning the Product under the\nConsumer Contracts Regulations\nand the\nPandora website enhanced cancellation policy\n(if applicable) have expired, please contact the Customer Care Team for further assistance, either by using our\nContact Us\nform, by telephone to the Freephone number 0808 2345 431) check\nopening hours here\nor by ordinary mail to Pandora at PFS Conexus, Unit 7 Mountpark Southampton Wide Lane, Southampton, SO18 2NQ.\nPandora will refund in full the price of Products (including sale Products) that are faulty or not as described, together with all applicable delivery charges (including any premium paid for non-standard delivery), and any reasonable costs you incur in returning the item.\nExchanges through the Pandora website\nProducts purchased from the Pandora website (including sale items) can also be exchanged in a Pandora store and certain Pandora stockists within 30 days from purchase, but only if you wish to exchange it for the same product in a different size or for a different product of exactly the same value. Please be aware that Pandora stockists have their own exchange policy which may differ to Pandora\'s, please contact the stockist directly for details.\nDue to a higher than usual volume of enquiries, we have temporarily suspended Product exchanges through the Pandora website. This means that you cannot exchange Products through the Pandora website. However, you may return Products to the Pandora website using the Consumer Contracts Regulations process set out above.\nExceptions\nIf you do not return a Product (or, if a Product sent using the free\nReturns Process\nfails to arrive and you are unable to provide proof of postage), Pandora reserves the right to refuse a refund.\n13. Promotions\nIf you participate in a Pandora promotion additional terms apply. For more information please read our\nPromotion FAQs\n.\n14. Consumer Warranty (Guarantee)\nTHIS GUARANTEE GIVES YOU SPECIFIC RIGHTS. IT IS OFFERED AS AN ADDITIONAL BENEFIT TO YOUR RIGHTS UNDER LAW. YOU ALSO HAVE OTHER RIGHTS UNDER LAW WHICH MAY VARY FROM COUNTRY TO COUNTRY.\nYou can find out about your legal rights and remedies by contacting your local consumer advice service. In particular, the Pandora website has a legal duty to supply products that conform to the contract made with you. Nothing in this Guarantee will replace or lessen any of your legal rights or remedies.\nProducts sold by the Pandora website include a guarantee for Products which turn out to be faulty or not as described during the periods set out below for various Product types:\n• 2 years from the original date of purchase for silver jewellery\n• 2 years from the original date of purchase for gold jewellery\n• 2 years from the original date of purchase for Pandora Rose and Pandora Shine jewellery\n• 1 year from the original date of purchase for parts of Products made of wood, leather, glass and string items.\nThis means you can request a repair or replacement free of charge if the Products turn out to be faulty or not as described during this period. Pandora will repair your Product or replace it with a Product of identical or similar characteristics, except as set out below. Pandora may at its own discretion replace your Product under this Guarantee whether or not a repair of your original Product is possible. If Pandora chooses instead to repair your Product Pandora may do so with new or refurbished components.\nIf the Product cannot be repaired or replaced within a reasonable time or without inconvenience, you may request a refund or price reduction. Refunds, if authorised, will usually be made using the same payment method originally used by you to pay for your purchase.\nPlease retain your Order Receipt Confirmation or pass it on to the gift recipient as the original proof of purchase is the warranty under which this Guarantee operates. If the Order Receipt Confirmation is not available, a credit card or bank statement showing the relevant transaction will suffice. This does not affect your rights in law as a consumer.\nThis Guarantee does not cover normal wear and tear or damage caused by accidents, mishandling, improper use (such as knocks, dents, crushing etc.).\nRepairs (or attempted repairs) to Products other than by Pandora will void this Guarantee whether or not damage has been caused by such repair or attempted repair.\nIf your Product is repaired or replaced by Pandora under this Guarantee, the new item will benefit from the remainder of the term of this Guarantee (calculated from the date of the original purchase of the Product). The period of this Guarantee shall not be extended whether or not your Product is repaired or replaced.\nWearing counterfeit charms, charms from incompatible Pandora collections or charms by other brands on your Pandora bracelet could damage the unique threading system and if such damage is caused, it will not be covered by this Guarantee.\n15. Liability\nPandora warrants to you that any Product purchased from the Pandora website is of satisfactory quality and fit for all the purposes for which products of the same kind are commonly supplied.\nPandora is only liable for foreseeable loss and damage. Pandora is not responsible for any loss or damage that is not foreseeable. HOWEVER NOTHING IN THESE SALES TERMS AFFECTS YOUR LEGAL RIGHTS AS A CONSUMER. TO THE FULL EXTENT PERMITTED BY LAW, THE TOTAL LIABILITY OF PANDORA AND PANDORA A/S SHALL BE LIMITED TO THE AMOUNT ACTUALLY PAID FOR THE PURCHASE OF THE APPLICABLE PRODUCTS PLUS DELIVERY FROM THE PANDORA WEBSITE.\nIn particular, nothing in these Sales Terms excludes or limits in any way the liability of Pandora:\n• for death or personal injury caused by Pandora\'s negligence;\n• for fraud or fraudulent misrepresentation;\n• under section 2(3) of the UK\'s Consumer Protection Act 1987 or for any breach of your consumer rights relating to products; or\n• for any matter for which it would be illegal for Pandora to limit or exclude, or attempt to limit or exclude, liability.\n16. Severability\nIf any provision of these Sales Terms shall be deemed unlawful, void or for any reason unenforceable, then that provision shall be deemed severable (deleted) from these Sales Terms and this will not affect the validity and enforceability of any remaining provisions.\n17. Contact Details\nPandora\'s\nCustomer Care Team\nwill assist you with any Pandora website order related questions or complaints. You can contact the Pandora Customer Care Team in the following ways:\n• via the\nContact Us\nform on the Pandora website;\n• by FREEPHONE 0808 2345 431 check\nopening hours here\nor\n*Please note, calls are free from any UK landline. UK mobile network charges might apply. Please check with your network provider.\nPandora will endeavour to resolve any complaint you may have. If Pandora is unable to resolve a dispute, either party is entitled to seek further recourse through the\nODR Platform\n. Operated by the European Commission, the ODR Platform is an online platform providing businesses and customers in the European Union with a forum for resolving online sales disputes without the need to go to court. The dispute resolution services available on the Platform are provided free of charge, although neither party is under any obligation to participate.\n18. Transferring rights and obligations to another person or organisation\nThese Sales Terms are binding on you and Pandora. It will also be binding on any person or organisation to which Pandora might transfer its rights and obligations to.\nYou may not transfer or otherwise dispose of your contract with Pandora, or any of your rights or obligations arising under it, without Pandora\'s prior written consent.\nPandora may transfer or otherwise dispose of a contract with you, or transfer any of Pandora\'s rights or obligations arising under it, at any time. If you have an outstanding order with Pandora at the time of any such disposal or transfer, you will be contacted if Pandora and you may cancel your outstanding order if you do not want to proceed.\n19. Third Party Rights\nOther than Pandora A/S, any person who is not a party to the contract between you and Pandora is not entitled to enforce any of its terms.\n20. Intellectual Property Rights\nThe content on the Pandora website including all jewellery design, copyright, trade marks and other intellectual property rights it contains, including the name Pandora is the sole property of Pandora or its licensors.\n21. Data Protection\nPandora believes strongly in protecting user privacy and personal data. Any personal data collected and processed by Pandora is governed by Pandora\'s\nPrivacy Policy\n. Please read Pandora\'s\nPrivacy Policy\nfor further information on how Pandora processes personal data collected from users of the Pandora website.\n22. Waiver\nIf Pandora fails, at any time, to insist that you comply with your obligations under these Sales Terms, or if Pandora does not exercise any of its rights under these Sales Terms, this does not constitute a waiver of such rights and does not mean that you are free to ignore your obligations. Pandora can still require you to comply at a later date.\n23. Entire Agreement\nThese Sales Terms and any document that we have referred to within them (including via hyperlink) represent the entire agreement between Pandora and you and takes precedence over any other previous agreement or representation made to you whether oral or in writing.\nYou acknowledge that, in entering into a contract with Pandora, you have not relied on any representation or promise given by Pandora or anyone else except as set out in these Sales Terms.\n24. Governing Law and Jurisdiction\nThese Sales Terms and your purchase of Products from Pandora through the Pandora website shall be governed by and construed in accordance with the laws of England.\nAny dispute or claim arising out of or in connection with these Sales Terms or your purchase of Products from Pandora (including non-contractual disputes or claims) shall be subject to the non-exclusive jurisdiction of the English courts. If you are a resident of Northern Ireland, Scotland or Wales you may also bring proceedings in your local courts\nLast updated\n15 June 2020"""
    # ins.main(SAMPLE_TEMPLATE)
    # text = ins.get_content_from_image(
    #     '/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/pdfpro/storage/dji-drone-invoice-template.png')
    # logger.info('\n'.join(text))
    # ins.generate_sample_by_sample('\n'.join(text))
    #     sample = """License Details
    # License #: 313807 Business Name: BENNETT ENGINEERING Status: Active Issue Date: 03/03/2006 Expiration Date: 12/31/2025 Has Telemedicine: No Mailing Address: 2825 CHIEF WILLIAM DRIVE, SUITE #12 FAIRBANKS, AK 99709 Physical Address: 2825 Chief William Dr, Unit 12
    # 9074795118
    # Fairbanks, AK 99709
    # Owners
    # F. LAWRENCE BENNETT, P.E.
    # Activities
    # Line of Business: 54 - Professional, Scientific and Technical Services
    # NAICS: 541330 - ENGINEERING SERVICES
    # Professional License # AELC1768
    # Line of Business: 54 - Professional, Scientific and Technical Services
    # NAICS: 541370 - SURVEYING AND MAPPING (EXCEPT GEOPHYSICAL) SERVICES
    # Professional License #: AELL3224
    # Endorsements
    # No Endorsements Found
    # License Lapse(s)
    # If this business license lapsed within the last four years the lapsed periods will appear below. Lapsed periods are the unlicensed period between an expiration date and renewal date.
    # No Lapses on record for the last 4 years."""
    #     sample = """2025-02-02T00:27:59+08:00 COMP6615226D {"time": "2025-02-01T16:27:59.3432132Z", "resourceId": "/tenants/bc0b541e-cf5f-48a5-a45d-7d041fe508a0/providers/Microsoft.aadiam", "operationName": "Sign-in activity", "operationVersion": "1.0", "category": "NonInteractiveUserSignInLogs", "tenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "resultType": "50097", "resultSignature": "None", "resultDescription": "Device Authentication Required - DeviceId -DeviceAltSecId claims are null OR no device corresponding to the device identifier exists.", "durationMs": 0, "callerIpAddress": "10.104.204.32", "correlationId": "3ad2b5b1-ef09-4c50-81ad-853aef57674d", "identity": "user_5f5d1039", "Level": 4, "location": "CN", "properties": {"id": "4d63c1c4-c95b-4d3c-8e0a-1271fd813e00", "createdDateTime": "2025-02-01T16:26:24.6955459+00:00", "userDisplayName": "user_5f5d1039", "userPrincipalName": "user_468aad99@domain_e9c4f53a.com", "userId": "6abee629-69f8-461c-9fab-71c4ff200168", "appId": "27922004-5251-4030-b22d-91ecd9a37ea4", "appDisplayName": "Outlook for iOS and Android", "ipAddress": "10.104.204.32", "status": {"errorCode": 50097, "failureReason": "Device Authentication Required - DeviceId -DeviceAltSecId claims are null OR no device corresponding to the device identifier exists.", "additionalDetails": "MFA requirement satisfied by claim in the token"}, "clientAppUsed": "Mobile Apps and Desktop clients", "userAgent": "Mozilla/5.0 (compatible; TESTAL 1.0) PKeyAuth/1.0", "deviceDetail": {"deviceId": "30a4051b-7be0-4754-a3b8-755991aa69e0", "displayName": "WS3A7C1DBA", "operatingSystem": "Ios 18.2.1", "trustType": "Azure AD registered"}, "location": {"city": "Chengdu", "state": "Sichuan", "countryOrRegion": "CN", "geoCoordinates": {"latitude": 30.653060913085938, "longitude": 104.06749725341797}}, "mfaDetail": {}, "correlationId": "3ad2b5b1-ef09-4c50-81ad-853aef57674d", "conditionalAccessStatus": "success", "appliedConditionalAccessPolicies": [{"id": "627ea70c-8186-40a5-aafb-d681edbc2d88", "displayName": "Block Access to O365 from Mobile Devices for All Users", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 7, "conditionsNotSatisfied": 0}, {"id": "b8944f3f-001d-4d7a-ad95-4b110f20a9f5", "displayName": "Block All Access from Untrusted IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "a72bcb04-325f-4428-8597-431bd40d390f", "displayName": "CA307 - Application  Access from IOS Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1039, "conditionsNotSatisfied": 0}, {"id": "a34fae20-bd1e-479e-95e8-23dbc7ad9064", "displayName": "Block Access to O365 from NON-COD IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "a2a977f6-8469-4a63-a26e-d33066160e17", "displayName": "Block MFA Registration from Untrusted IP", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 0}, {"id": "c061f6ef-16a9-4939-96ff-96b4de4b2de5", "displayName": "Block Exchange Online Access from BYO IOS", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "failure", "conditionsSatisfied": 263, "conditionsNotSatisfied": 0}, {"id": "61e7500a-54bc-4a7b-9da8-d97ca911b485", "displayName": "Enable Exchange Online Access from Managed IOS Device", "enforcedGrantControls": ["RequireCompliantDevice", "RequireCompliantApp"], "enforcedSessionControls": ["SignInFrequency"], "result": "failure", "conditionsSatisfied": 23, "conditionsNotSatisfied": 0}, {"id": "fc6f39dc-08c0-45af-b52e-053e1a9e5dd8", "displayName": "AADC DirSync Account Restriction", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "31bd8466-c9a4-49da-9bfd-68bb5ea30c48", "displayName": "Enable MFA for Micorosft Azure Management - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "4f94a30e-8595-4522-8594-a7b38dcf7de7", "displayName": "Enable MFA for O365 - AAD Roles", "enforcedGrantControls": ["COD PA Training"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "907260ac-d47d-4cb1-a266-3f757b3c3d27", "displayName": "Enable MFA for CyberArk - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "91f9176a-3641-4c97-af5c-c02ee23bb741", "displayName": "Enable SIF for All Services from Intranet - 7 Days", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 3, "conditionsNotSatisfied": 1032}, {"id": "efddddee-44c8-435f-b661-60d11515a36b", "displayName": "Enable SIF for CyberArk - 12 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "476d40ea-446e-497b-913a-497e7c8464b2", "displayName": "Enable SIF for Cloud BTG Accounts - 4 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 1, "conditionsNotSatisfied": 2}, {"id": "e8a6b9ec-85c4-4948-afb8-544ce8202f89", "displayName": "Enable SIF for Critical Services - 12 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "986eb6ae-1169-4c2e-9194-a2317222a4d3", "displayName": "Block Access to Power Platform for All Users", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "179fc5f0-50e0-4594-9efb-fb07f8e75aba", "displayName": "CA301 - Register or Join Device \\u2013 MFA ", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "0fa2d843-684c-4487-92e1-8707576c26b9", "displayName": "CA306 - Application External Access from Linux Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "3d01b038-5495-4cbe-913b-08dd47898438", "displayName": "CA305 - Application External Access from Windows Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "14bab089-a460-4838-987c-1dfa7b0173c5", "displayName": "CA303 - Company Portal App External Login on Unmanaged Device", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "bcd46537-f715-4fa6-b9ca-0cd996f34571", "displayName": "CA304 - Application Access from OtherOS Device - Block", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 1035, "conditionsNotSatisfied": 4}, {"id": "3c38edf0-c14c-4b44-a85a-69118bb366ec", "displayName": "CA302 - Company Portal App External Login on Managed Device - MFA", "enforcedGrantControls": ["RequireCompliantDevice"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "332f46ad-ce38-4620-b757-a3445cb0bd1b", "displayName": "Block Restricted User For Citrix Access", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "8842877d-af5a-4e9c-9b65-405486a5a4bf", "displayName": "Enable MFA for Citrix Access", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "549eb90d-2f20-469b-a29b-4fb1d7204991", "displayName": "Enable SIF for Citrix Access - Every Time", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "c8e61b40-6a11-4d07-bc8a-6dcf5db87fe1", "displayName": "Enable Regular PA Training for PA users", "enforcedGrantControls": ["Mfa", "COD PA Training"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "cc67f465-32a1-4d6e-9901-7b95d7709396", "displayName": "Block Exchange Online Access from Browser of Managed IOS Device", "enforcedGrantControls": ["Block"], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 7, "conditionsNotSatisfied": 16}, {"id": "0f959fe2-497a-4849-aad8-1a50abbc837e", "displayName": "Enable MFA for ReversingLabs Software Assurance Managed Service", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "e548b13e-f481-44ba-86e4-ee94f6fa5d49", "displayName": "Enable MFA for CNAPS2 - All Users", "enforcedGrantControls": [], "enforcedSessionControls": [], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}, {"id": "cd5e57d9-7085-4099-b7f4-1c5622aee6d3", "displayName": "Enable SIF for CNAPS2 - 4 Hours", "enforcedGrantControls": [], "enforcedSessionControls": ["SignInFrequency"], "result": "notApplied", "conditionsSatisfied": 0, "conditionsNotSatisfied": 1}], "authenticationContextClassReferences": [{"id": "urn:user:registersecurityinfo", "detail": "required"}, {"id": "urn:user:registerdevice", "detail": "previouslySatisfied"}], "originalRequestId": "4d63c1c4-c95b-4d3c-8e0a-1271fd813e00", "isInteractive": false, "tokenIssuerName": "", "tokenIssuerType": "AzureAD", "authenticationProcessingDetails": [{"key": "Legacy TLS (TLS 1.0, 1.1, 3DES)", "value": "False"}, {"key": "Is CAE Token", "value": "False"}], "networkLocationDetails": [], "clientCredentialType": "none", "processingTimeInMilliseconds": 133, "riskDetail": "none", "riskLevelAggregated": "none", "riskLevelDuringSignIn": "none", "riskState": "none", "riskEventTypes": [], "riskEventTypes_v2": [], "resourceDisplayName": "Microsoft Graph", "resourceId": "00000003-0000-0000-c000-000000000000", "resourceTenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "homeTenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "tenantId": "a2164b8e-d052-4f10-8352-fa46f761a964", "authenticationDetails": [], "authenticationRequirementPolicies": [], "sessionLifetimePolicies": [], "authenticationRequirement": "singleFactorAuthentication", "servicePrincipalId": "", "userType": "Member", "flaggedForReview": false, "isTenantRestricted": false, "autonomousSystemNumber": 4134, "crossTenantAccessType": "none", "privateLinkDetails": {}, "ssoExtensionVersion": "", "uniqueTokenIdentifier": "xMFjTVvJPE2OChJx_YE-AA", "authenticationStrengths": [], "incomingTokenType": "refreshToken", "authenticationProtocol": "none", "appServicePrincipalId": null, "resourceServicePrincipalId": "b636ca69-c274-4271-8ca7-1649a6ab81ba", "rngcStatus": 0, "signInTokenProtectionStatus": "none", "tokenProtectionStatusDetails": {"signInSessionStatus": "unbound", "signInSessionStatusCode": 1004}, "originalTransferMethod": "none", "isThroughGlobalSecureAccess": false, "conditionalAccessAudiences": [{"applicationId": "00000003-0000-0ff1-ce00-000000000000", "audienceReasons": "none"}, {"applicationId": "00000002-0000-0ff1-ce00-000000000000", "audienceReasons": "none"}, {"applicationId": "00000002-0000-0000-c000-000000000000", "audienceReasons": "none"}, {"applicationId": "ea890292-c8c8-4433-b5ea-b09d0668e1a6", "audienceReasons": "none"}], "sessionId": "811fe6e4-f4bb-4684-8978-5ce5de39f8b9", "resourceOwnerTenantId": "28bd4fd2-a6ec-43cc-a6ab-416b4ef214f3"}}
    # 2025-02-01T23:30:33+08:00 COMPE3FDB04F <Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Microsoft-Windows-Security-Auditing' Guid='{54849625-5478-4994-a5ba-3e3b0328c30d}'/><EventID>4625</EventID><Version>0</Version><Level>0</Level><Task>12544</Task><Opcode>0</Opcode><Keywords>0x8010000000000000</Keywords><TimeCreated SystemTime='2025-02-01T15:30:33.499045800Z'/><EventRecordID>6858529940</EventRecordID><Correlation/><Execution ProcessID='1376' ThreadID='12412'/><Channel>Security</Channel><Computer>COMPE3FDB04F</Computer><Security/></System><EventData><Data Name='SubjectUserSid'>NT AUTHORITY\\\\SYSTEM</Data><Data Name='SubjectUserName'>BJVMWCDC01$</Data><Data Name='SubjectDomainName'>COD</Data><Data Name='SubjectLogonId'>0x3e7</Data><Data Name='TargetUserSid'>NULL SID</Data><Data Name='TargetUserName'>ltm-bind</Data><Data Name='TargetDomainName'>COD</Data><Data Name='Status'>0xc000006d</Data><Data Name='FailureReason'>%%2313</Data><Data Name='SubStatus'>0xc000006a</Data><Data Name='LogonType'>3</Data><Data Name='LogonProcessName'>Advapi  </Data><Data Name='AuthenticationPackageName'>MICROSOFT_AUTHENTICATION_PACKAGE_V1_0</Data><Data Name='WorkstationName'>WS4CCADB6B</Data><Data Name='TransmittedServices'>-</Data><Data Name='LmPackageName'>-</Data><Data Name='KeyLength'>0</Data><Data Name='ProcessId'>0x560</Data><Data Name='ProcessName'>C:\\\\Windows\\\\System32\\\\lsass.exe</Data><Data Name='IpAddress'>10.134.159.249</Data><Data Name='IpPort'>38058</Data></EventData></Event>"""
    #     img_path = Path(r"/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/pdfpro/storage/dior-receipt-template.png")
    #     remake_sample = ins.main(SAMPLE_TEMPLATE, sample_count=2)
    #     print(remake_sample)
    files = glob.glob(
        "/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/mailcharts/storage/order_confirmation/*")
    # for file in tqdm.tqdm(files):
    #     with open(file, 'r') as f:
    #         data = f.read()
    #     st = time.time()
    #     ins.generate_sample_by_html_p_extraction(data, Path(file).stem)
    #     logger.success(f"Used {time.time() - st} seconds.")

    # 创建进程池，根据你的CPU核心数决定同时运行的进程数量
    num_processes = multiprocessing.cpu_count()  # 或者指定一个具体的数值
    pool = multiprocessing.Pool(processes=num_processes)

    # 使用map方法并行处理所有文件，并收集结果
    results = list(tqdm.tqdm(pool.imap_unordered(process_file, files), total=len(files)))

    # 关闭进程池，等待所有进程完成
    pool.close()
    pool.join()

    # 如果你想记录每个文件的处理时间，可以在这里遍历results
    for file, elapsed_time in results:
        logger.success(f"Processing {file} used {elapsed_time} seconds.")