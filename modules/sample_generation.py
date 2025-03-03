import glob
import os
import time

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
        seed_path = batch_dir/'seed_content.txt'
        with open(seed_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        pii_extracted = self.pii_extractor.main(pii_category='general', input_str=sample_content, votes=5)
        pre_text, placeholder_maps, person_mapping = self.generate_sample_by_sample_p_extraction(sample_content,
                                                                                                 pii_extracted)
        sample_path = batch_dir/'sample_content.txt'
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
        logger.debug(extracted_piis)
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
        template, instance_count, pii_label_to_pii_content_mapping = self.sample_to_template(sample_text, extracted_piis=pii_extracted)
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

    def generate_sample_by_html_p_extraction(self, sample_text, storage_name=None):
        if storage_name is None:
            storage_name = str(int(time.time()*1000))
        logger.info(f"Name: {storage_name}")
        if (Path(__file__).parent / 'ecommerce_outs' / f'{storage_name}.html').exists():
            logger.success(f"{storage_name} Already finished.")
            return
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(sample_text, 'lxml')

        # 提取所有文本节点，排除纯空白或仅包含空白字符的文本
        texts = [element.get_text(strip=True) for element in soup.find_all(text=True) if element.get_text(strip=True)]
        text_all = '\n'.join(texts)
        pii_extracted = self.pii_extractor.main(pii_category='ecommerce', input_str=text_all, votes=5)
        template, instance_count, pii_label_to_pii_content_mapping = self.sample_to_template(text_all, extracted_piis=pii_extracted)
        logger.info(f"Generated template is: \n{template}")
        # Refine sample
        person_mapping = {str(i): PersonGenerator() for i in range(instance_count)}
        line_mapping = {}
        generated_content = None
        for attempt in range(5):
            content, reason = self.fill_template_by_llm(template, person_mapping)
            content = re.sub(r"\n+", "\n", content)
            logger.info(f"Content Generated: \n{content}")
            logger.info(f"\nComments: {reason}")
            text_all = re.sub(r"\n+", "\n", text_all)
            logger.info(f"Prev content: \n{text_all}")
            if len(list(text_all.split('\n'))) != len(list(content.split('\n'))):
                len_text_all = len(list(text_all.split('\n')))
                len_content = len(list(content.split('\n')))
                logger.warning(f"Generated content line count doesnt match {len_text_all} != {len_content} . Generate again.")
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
            os.makedirs(Path(__file__).parent/'ecommerce_outs', exist_ok=True)
            with open(Path(__file__).parent/'ecommerce_outs'/f'{storage_name}.html', 'w') as f:
                f.write(html_sample)
            logger.success(f"Sample stored at {Path(__file__).parent/'ecommerce_outs'/f'{storage_name}.html'}")


    # @staticmethod
    # def extract_filled_content(source_text, pii_label_to_pii_content_mapping, final_text):
    #     # Determine uploaded info
    #     filled_content_mapping = {}
    #     for pii_label in pii_label_to_pii_content_mapping:
    #         logger.info(f"Differentiating {pii_label} texts")
    #         filled_content_mapping[pii_label] = []
    #         for text in pii_label_to_pii_content_mapping[pii_label]:
    #             if text not in source_text:
    #                 logger.warning(f"Tried to determine {pii_label}:{text} however not in pre_text. Skipped.")
    #                 continue
    #             prev_content_split = re.search(f'[\s|^](\S+{re.escape(text)})\S+', source_text).group(0)
    #
    #             # logger.info(f"Try to generate REGEX for {text}")
    #             # regex = re.sub(rf'{re.escape(text)}',
    #             #                f"REGEX",
    #             #                source_text)
    #             # regex = re.sub('REGEX', '(.*?)', re.escape(regex))
    #             # logger.info(f"Tried to use Regex: \n{regex}")
    #             # match = re.search(regex, re.escape(final_text))
    #             # if not match:
    #             #     logger.error(f"Failed to find synthesized item for {pii_label}: {text}")
    #             #     continue
    #             # filled_content = match.group(1)
    #             filled_content_mapping[pii_label].append(filled_content)
    #     return filled_content_mapping

    def fill_template_by_llm(self, pre_text, person_mapping):
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'template_fill_by_LLM.prompt'
        person_mapping_part = ["Here are person and corresponding personal Information related to the context."]
        for person_id in person_mapping:
            person_notation = "main_person" if person_id == '' else f"person_{person_id}"
            person_info = person_mapping[person_id].summary()
            person_mapping_part.append(f"{person_notation} name is: {person_info['FullName']}: \n{str(person_info)}")
        person_mapping_string = '\n'.join(person_mapping_part)

        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        parser = PydanticOutputParser(pydantic_object=RefineResult)
        model_instance = self.create_llm_instance(temperature=0, model_name='zhipu_glm4_plus')

        prompt = template.format(format_instruction=parser.get_format_instructions(),
                                 # pii_mappings=pii_mappings,
                                 document_content=pre_text,
                                 person_mapping_string=person_mapping_string)
        # logger.debug(prompt)
        res_content = model_instance.invoke(prompt)
        logger.warning(f"{res_content.model_dump_json()}")
        answer = parser.parse(res_content.content)
        logger.success(answer)

        return answer.refined_context, answer.reason

    def main(self, input_body, batch_dir=None, sample_count=1):
        if not batch_dir:
            batch_dir = Path(__file__).parent/'output'/str(time.time()*1000000)
        if isinstance(batch_dir, str):
            batch_dir = Path(batch_dir)
        logger.info(f"STARTS TO GENERATE SAMPLE AT {batch_dir}")
        input_type = self.determine_input_type(input_body)
        logger.info(f"Input body is {input_type}")

        if input_type == "TEMPLATE":
            # content, mapping, reason = self.generate_sample_by_template(input_body)
            seed_document, template_body, synthetic_content = self.generate_sample_by_template(input_body, sample_count=sample_count)
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
    ins = SampleGeneration()
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
    files = glob.glob("/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/mailcharts/storage/order_confirmation/*")

    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            data = f.read()
        ins.generate_sample_by_html_p_extraction(data, Path(file).stem)