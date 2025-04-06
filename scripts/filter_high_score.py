import glob
import json
import os
import random
import re
import shutil
import string
import time
import traceback
import hashlib

from modules.pii_generators.person_pii_generator import PersonGenerator
import tqdm
from loguru import logger
from pathlib import Path
from typing import List, Dict
from modules.PII_extraction import PIIExtraction
from modules.sample_generation import SampleGeneration
from multiprocessing import Pool, cpu_count

SAMPLE_BASE = Path('/output/ecommerce_v0')
SCORE_REQUIREMENT = 7
FILTERED_PATH = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/good_samples")
GOLD_DELIVERY_HOME = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home')
GOLD_SAMPLES_HOME = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/good_samples')
GOLD_TEMPLATES_HOME = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/good_templates')


def convert_paths_to_strings(d):
    if isinstance(d, dict):
        return {k: convert_paths_to_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_paths_to_strings(element) for element in d]
    elif isinstance(d, Path):
        return str(d)
    else:
        return d


def get_score(sample_content_path: Path):
    sample_score_path = sample_content_path.parent / str(
        sample_content_path.stem + "_score.json")
    if sample_score_path.exists():
        with open(sample_score_path, 'r') as f:
            sample_score_data = json.load(f)
        score = sample_score_data.get('final_score')
        reason = sample_score_data.get('reason')
    else:
        score = 0
        reason = None
    return score, reason


def get_high_score_samples():
    batch_with_template = glob.glob(str(SAMPLE_BASE / '**' / '*template.txt'))
    todo_batches = [Path(i).parent for i in batch_with_template]
    all_sample_with_high_score = []
    for batch_dir in todo_batches:
        task_status_files = glob.glob(str(batch_dir / 'batch_status*.json'))
        samples = []
        pii_category = 'ecommerce_v0'
        for task_status_file in task_status_files:
            try:
                with open(task_status_file, 'r') as f:
                    task_status = json.load(f)
                samples.extend(task_status.get('generated_samples', []))
                pii_category = task_status.get('pii_category') if task_status.get('pii_category') else pii_category
            except Exception as e:
                logger.error(f"STATUS JSON {task_status_file} OPEN FAILED: {traceback.print_exc()}")
                continue
        for gen_sample in samples:
            if not gen_sample:
                continue
            score, reason = get_score(Path(gen_sample['sample_content']))

            if score >= SCORE_REQUIREMENT:
                # logger.info(f"GOOD SAMPLE: {Path(gen_sample['sample_content'])}")
                all_sample_with_high_score.append(gen_sample)
    logger.info(f"{len(all_sample_with_high_score)} GOOD SAMPLES")
    return all_sample_with_high_score


def get_low_score_samples():
    batch_with_template = glob.glob(str(SAMPLE_BASE / '**' / '*template.txt'))
    todo_batches = [Path(i).parent for i in batch_with_template]
    all_sample_with_bad_score = []
    for batch_dir in todo_batches:
        task_status_files = glob.glob(str(batch_dir / 'batch_status*.json'))
        samples = []
        pii_category = 'ecommerce_v0'
        for task_status_file in task_status_files:
            try:
                with open(task_status_file, 'r') as f:
                    task_status = json.load(f)
                samples.extend(task_status.get('generated_samples', []))
                pii_category = task_status.get('pii_category') if task_status.get('pii_category') else pii_category
            except Exception as e:
                logger.error(f"STATUS JSON {task_status_file} OPEN FAILED: {traceback.print_exc()}")
                continue
        for gen_sample in samples:
            if not gen_sample:
                continue
            score, reason = get_score(Path(gen_sample['sample_content']))

            if score < SCORE_REQUIREMENT:
                all_sample_with_bad_score.append(gen_sample)
    logger.info(f"{len(all_sample_with_bad_score)} Bad SAMPLES")
    return all_sample_with_bad_score


def get_high_score_templates_batch_paths(samples=None):
    if not samples:
        sample_dicts = get_high_score_samples()
        samples = [i['sample_content'] for i in sample_dicts]
    templates = set()
    for sample in samples:
        template_path = Path(sample).parent
        templates.add(template_path)
    logger.info(f"{len(templates)} GOOD TEMPLATES")
    return templates


def get_low_score_templates(samples=None):
    if not samples:
        sample_dicts = get_low_score_samples()
        samples = [i['sample_content'] for i in sample_dicts]
    templates = set()
    for sample in samples:
        template_path = Path(sample).parent
        templates.add(template_path)
    logger.info(f"{len(templates)} BAD TEMPLATES")
    return templates


def deliver_selected_templates_results(template_paths: List[Path]):
    delivery_path = GOLD_DELIVERY_HOME / f'delivery_{str(int(time.time()))}'
    for template_path in template_paths:
        pass


def deliver_selected_samples(sample_dicts: List[Dict]):
    delivery_path = GOLD_DELIVERY_HOME / f'delivery_{str(int(time.time()))}'
    os.makedirs(delivery_path, exist_ok=True)
    pii_category_config = PIIExtraction().load_config('ecommerce_v0')
    pii_categories = [i['key_name'] for i in pii_category_config[0]]
    to_remove = ['OrderNumber', 'SocialSecurityNumber', 'Initials', 'PassportNumber']
    for i in to_remove:
        pii_categories.remove(i)

    out = []
    for sample_dict in tqdm.tqdm(sample_dicts):
        out_dict = {}
        sample_content_path = sample_dict.get('sample_content')
        extracted_piis_all = []
        with open(sample_content_path, 'r') as f:
            sample_content = f.read()
        out_dict['sample_content'] = sample_content
        sample_pii_extracted_path = Path(sample_content_path).parent / str(
            Path(sample_content_path).stem + '_pii_extraction.json')
        if sample_pii_extracted_path.exists():
            with open(sample_pii_extracted_path, 'r') as f:
                extracted_piis = json.load(f)
            logger.success("Already extracted PIIs")
        else:
            extracted_piis = []
        if len(extracted_piis) < 4:
            continue
        if len(extracted_piis) >= 20:
            continue
        for r in extracted_piis:
            if r['pii_class'] in pii_categories and r['pii_content'] in sample_content:
                extracted_piis_all.append({r['pii_content']: r['pii_class']})
        out_dict['extracted_piis'] = extracted_piis_all
        out_dict['sample_pii_extracted_path'] = str(sample_pii_extracted_path)
        placeholder_content_map = sample_dict.get('placeholder_content_map')
        with open(placeholder_content_map, 'r') as f:
            placeholder_content_map_data = json.load(f)
        placeholder_piis = []
        if placeholder_content_map_data:
            p_c_mapping = {}
            for p_c in placeholder_content_map_data:
                try:
                    p_c_mapping.update(p_c)
                except:
                    logger.warning(p_c)
                    logger.error(traceback.print_exc())
            for p_name in pii_categories:
                for placeholder in p_c_mapping:
                    if p_name in placeholder:
                        placeholder_piis.append({p_c_mapping[placeholder]: p_name})
        out_dict['placeholder_piis'] = placeholder_piis
        out.append(out_dict)
        with open(delivery_path / 'delivery_summary.json', 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


def unit_generate_selected_template(template_path: Path, out_path: Path):
    generator = SampleGeneration()
    os.makedirs(out_path, exist_ok=True)
    batch_details = generator.generate_sample_by_template(raw_template_path=template_path,
                                                          batch_dir=out_path,
                                                          sample_count=1,
                                                          template_only=False)
    return batch_details


def write_to_summary(summary_path: Path, batch_details):
    """线程安全地向summary文件追加数据"""
    with open(summary_path, 'a', encoding='utf-8') as f:
        json.dump(batch_details, f, ensure_ascii=False)
        f.write('\n')


def handle_batch_template_gen(template_path: Path, out_path: Path):
    template_result_batch_path = out_path / template_path.name
    batch_details = unit_generate_selected_template(template_path, template_result_batch_path)
    return convert_paths_to_strings(batch_details)


def generate_selected_templates(template_paths: List[Path], multiprocess=False):
    out_path = GOLD_SAMPLES_HOME / f'gold_samples_{str(int(time.time()))}'
    summary_path = out_path / 'summary.json'
    os.makedirs(out_path)
    if not multiprocess:
        summary = []
        for template_path in template_paths:
            template_result_batch_path = out_path / template_path.stem
            batch_details = unit_generate_selected_template(template_path, template_result_batch_path)
            summary.append(batch_details)
            with open(summary_path, 'w') as f:
                json.dump(convert_paths_to_strings(summary), f, ensure_ascii=False, indent=4)

    else:
        # 初始化summary文件
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('[]')  # 创建一个空的JSON数组

        num_processes = cpu_count()
        num_processes = num_processes // 2  # 根据CPU核心数自动选择进程数
        logger.info(f"Using {num_processes} cores.")
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(handle_batch_template_gen,
                                   [(template_path, out_path) for template_path in template_paths])

            # 将每个结果写入summary文件
            for batch_details in tqdm.tqdm(results, desc="Processing templates"):
                write_to_summary(summary_path, batch_details)


def relocate_templates(batch_paths: List[Path], dst_folder: Path):
    out = {}
    for p in tqdm.tqdm(batch_paths):
        template_path = glob.glob(str(p / '*template.txt'))
        if template_path:
            template_path = template_path[0]
            dst_path = dst_folder / f"{p.stem}.txt"
            shutil.copy(template_path, dst_path)
            out[p] = dst_path
        else:
            logger.error(f"Failed to find template: {p}")
            continue
    return out


def delivery_shuffle_number(delivery_summary_path: Path):
    modified_path = delivery_summary_path.parent / str(delivery_summary_path.stem + f'_{str(int(time.time()))}.json')
    with open(delivery_summary_path, 'r') as f:
        data = json.load(f)

    new_out = []

    for entry in tqdm.tqdm(data):
        sample_content = entry['sample_content']
        labels = entry['extracted_piis']
        masked_sample_content = sample_content
        mask_map = {}
        number_shuffle_map = {}
        for idx, label in enumerate(labels):
            label_content = list(label.keys())[0]
            masked_sample_content = masked_sample_content.replace(label_content, f"[MASK_{'A'*idx}]")
            mask_map[f"[MASK_{'A'*idx}]"] = label_content

        number_extracts = re.findall('(\d{3,})', masked_sample_content)
        number_extracts = list(set(number_extracts))

        for number in number_extracts:
            new_number = str(random.randint(10 ** (len(number) - 1), 10 ** (len(number)) - 1))
            number_shuffle_map[new_number] = number
            masked_sample_content = masked_sample_content.replace(number, new_number)
        # for idx, label in enumerate(labels):
        #     label_content = list(label.keys())[0]
        #     masked_sample_content = masked_sample_content.replace(f"[MASK_{alpha[idx]}]", label_content)
        for mask in mask_map.keys():
            masked_sample_content = masked_sample_content.replace(mask, mask_map[mask])
        if "MASK" in masked_sample_content:
            print(masked_sample_content)
            print(mask_map)
        entry['sample_content'] = masked_sample_content
        entry['replace_map'] = number_shuffle_map
        new_out.append(entry)
    with open(modified_path, 'w') as f:
        json.dump(new_out, f, ensure_ascii=False, indent=2)


def unit_pii_extraction(pii_category, sample_content):
    instance = PIIExtraction()
    extracted_piis = instance.main(pii_category=pii_category,
                                   input_str=sample_content,
                                   votes=5)
    return extracted_piis


def process_item(data):
    pii_category_config = PIIExtraction().load_config('ecommerce_v0')
    pii_categories = [i['key_name'] for i in pii_category_config[0]]
    pii_categories.remove('OrderNumber')
    normal_extraction_path = data.get("sample_pii_extracted_path")
    openai_extraction_path = Path(normal_extraction_path).parent / f"{Path(normal_extraction_path).stem}_openai_35.json"

    extracted_piis_all = []

    if not openai_extraction_path.exists():
        # with get_openai_callback() as cb:
        #     res = unit_pii_extraction('ecommerce_v0', data.get('sample_content'))
        #     with open(openai_extraction_path, 'w') as f:
        #         json.dump(res, f, indent=2, ensure_ascii=False)
        # data['openai35_tokens'] = cb.total_tokens
        extracted_piis = []
    else:
        logger.success("Already openai35 extracted.")
        with open(openai_extraction_path, 'r') as f:
            extracted_piis = json.load(f)
    for r in extracted_piis:
        if r['pii_class'] in pii_categories and r['pii_content'] in data.get('sample_content'):
            extracted_piis_all.append({r['pii_content']: r['pii_class']})
    data['openai_35_pii_extracted'] = extracted_piis_all
    data['openai_35_sample_pii_extracted_path'] = str(openai_extraction_path)

    return data


def delivery_openai_extraction(delivery_summary_path: Path, multiprocess=False):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_openai35_extracted.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    out = []
    if multiprocess:
        num_processes = cpu_count()  # 根据CPU核心数自动选择进程数
        with Pool(processes=num_processes) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(process_item, datas), total=len(datas)))
            out.extend(results)
    else:
        for data in tqdm.tqdm(datas):
            result = process_item(data)
            out.append(result)

    with open(new_delivery_summary_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_sample_expand(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_openai35_extended.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    ins = SampleGeneration()
    if new_delivery_summary_path.exists():
        with open(new_delivery_summary_path, 'r') as f:
            out = json.load(f)
    else:
        out = []

    already_generated_ones = [i['seed_content'] for i in out]

    for data in tqdm.tqdm(datas):
        sample_content = data['sample_content']
        if sample_content in already_generated_ones:
            logger.success("Already Generated")
            continue
        res = ins.generate_sample_by_sample_totally_random(sample_content)
        cur = {'sample_contents': res,
               'seed_content': sample_content}
        logger.success(json.dumps(cur, indent=2, ensure_ascii=False))
        out.append(cur)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_check_high_freq_items(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_altered_most_freq.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    all_pii_contents = {}
    for data in datas:
        absolute_res = data.get('absolute_correct')
        absolute_map = {}
        for i in absolute_res:
            absolute_map.update(i)
        for k in absolute_map:
            pii_content = k
            pii_label = absolute_map[k]
            if pii_label not in all_pii_contents:
                all_pii_contents[pii_label] = {}
            if pii_content.upper() not in all_pii_contents[pii_label]:
                all_pii_contents[pii_label][pii_content.upper()] = 0
            all_pii_contents[pii_label][pii_content.upper()] += 1
    print(json.dumps(all_pii_contents, indent=2, ensure_ascii=False))
    dangerous_list = {}
    for pii_label in all_pii_contents:
        threshold_map = {'PhoneNumber': 1,
                         'Timestamps': 1,
                         'City': 10,
                         'CreditCardNumber': 1,
                         'Date': 4,
                         'StateAbbreviation': 100000,
                         'State': 100000,
                         'Country': 100000,
                         'FirstName': 100,
                         'LastName': 100,
                         'UserName': 1,
                         'Initials': 5}
        dangerous_list[pii_label] = [i for i in all_pii_contents[pii_label].keys() if
                                     all_pii_contents[pii_label][i] > 1]


def delivery_check_absolute_items(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_absolute.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    out = []
    for data in tqdm.tqdm(datas):
        zhipu_res = data.get('extracted_piis', [])
        openai_res = data.get('openai_35_pii_extracted', [])
        placeholder_res = data.get('placeholder_piis', [])
        zhipu_map = {}
        for i in zhipu_res:
            zhipu_map.update(i)
        openai_map = {}
        for i in openai_res:
            openai_map.update(i)
        zhipu_key = [list(i.keys())[0] for i in zhipu_res]
        openai_key = [list(i.keys())[0] for i in openai_res]
        placeholder_key = [list(i.keys())[0] for i in placeholder_res]

        absolute = []

        if not openai_key:
            for k in zhipu_key:
                if k in placeholder_key:
                    absolute.append({k: zhipu_map.get(k)})
        else:
            for k in zhipu_key:
                if zhipu_map[k] == openai_map.get(k):
                    absolute.append({k: zhipu_map[k]})
                else:
                    if k in placeholder_key:
                        absolute.append({k: zhipu_map[k]})
                    elif not placeholder_key:
                        absolute.append({k: zhipu_map[k]})
            for k in openai_key:
                if openai_map[k] == zhipu_map.get(k):
                    absolute.append({k: openai_map[k]})
                else:
                    if k in placeholder_key:
                        absolute.append({k: openai_map[k]})
        data['absolute_correct'] = absolute
        out.append(data)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def match_case(original, replacement):
    """
    根据原始单词的大小写格式调整替换单词的大小写。
    """
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.capitalize()
    if original.isupper():
        return replacement.upper()
    return replacement


def replace_keep_case(pattern, replacement, text):
    def _replace(match):
        matched_text = match.group(0)
        return match_case(matched_text, replacement)

    return re.sub(pattern, _replace, text)


def delivery_sub_stupid_stuff(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_subbed.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)

    SUB_DICT = ['lorem ipsum', 'loremipsum', 'lorem', 'ipsum', 'mailcharts', 'mail', 'charts', 'W CERMAK RD', 'fl 2',
                'c/o Simon\n']
    SUB_DICT = [i.upper() for i in SUB_DICT]
    A_DICT = ['FullName', 'FullName', 'FirstName', 'LastName', 'FirstName', 'FirstName', 'LastName', 'FirstName',
              'StreetName', 'StreetName']

    final_out = []
    for data in tqdm.tqdm(datas):
        labels = data.get('absolute_correct')
        if not labels:
            labels = data.get("extracted_piis")
        label_map = {}
        for l in labels:
            label_map.update(l)
        finished_label_map = {}
        content_sub_map = {}
        for key in label_map:
            found = 0
            for i in SUB_DICT:
                if i in key.upper():
                    person = PersonGenerator()
                    person.if_middle_name = 0
                    if label_map[key] == 'FirstName':
                        new_key = key.upper().replace(i, person.FirstName.upper())
                        finished_label_map[new_key] = label_map[key]
                        content_sub_map[key] = new_key
                    elif label_map[key] == 'LastName':
                        new_key = key.upper().replace(i, person.LastName.upper())
                        finished_label_map[new_key] = label_map[key]
                        content_sub_map[key] = new_key
                    elif label_map[key] == 'FullName':
                        new_key = key.upper().replace(i, person.FullName.upper())
                        finished_label_map[new_key] = label_map[key]
                        content_sub_map[key] = new_key
                    else:
                        pass
                    found = 1
                    break
            if not found:
                finished_label_map[key] = label_map[key]
        new_content = data['sample_content']
        for k in content_sub_map.keys():
            # print(k, content_sub_map[k])
            new_content = new_content.replace(k, content_sub_map[k])
        person = PersonGenerator()
        for k, a in zip(SUB_DICT, A_DICT):
            if k.upper() not in new_content.upper():
                continue
            pattern = re.compile(rf'\b({k})\b', re.IGNORECASE)
            replacement = person.__getattribute__(a)
            # print(replacement)
            new_content = replace_keep_case(pattern, replacement, new_content)
            # print(new_content)
        new_data = {'sample_content': new_content,
                    'piis': finished_label_map}
        final_out.append(new_data)
    with open(new_delivery_summary_path, 'w') as f:
        json.dump(final_out, f, indent=2, ensure_ascii=False)


def delivery_only_extract_address(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_add_addressONLY.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    ins = PIIExtraction()
    out = []
    for data in tqdm.tqdm(datas):
        content = data['sample_content']
        piis = ins.main('ecommerce_v0', content, 1,
                        ['StreetNumber', 'StreetName', 'ZipCode', 'City', 'Country', 'State', 'StateAbbreviation'])
        data['address_piis'] = piis
        out.append(data)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_only_full_address(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_add_FULL_addressONLY.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    ins = PIIExtraction()
    out = []
    for data in tqdm.tqdm(datas):
        content = data['sample_content']
        try:
            piis = ins.extract_specific(['Address'],
                                        ['Any forms of address. Including street name, street number, block id, building name, etc. ADDRESS ONLY.'],
                                        content)
        except:
            piis = []
        data['full_address_piis'] = piis
        out.append(data)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_only_names(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_add_address.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    ins = PIIExtraction()
    if new_delivery_summary_path.exists():
        with open(new_delivery_summary_path, 'r') as f:
            out = json.load(f)
    else:
        out = []

    for data in tqdm.tqdm(datas):
        content = data['sample_content']
        if content in [i['sample_content'] for i in out]:
            if data.get('name_only'):
                logger.success("ALEASY")
                continue
        # # try:
        # piis = ins.extract_specific(['FirstName', 'LastName', 'FullName', 'UserName'],
        #                             ['first name of people mentioned',
        #                              'last name of people mentioned.',
        #                              'full name of people mentioned.',
        #                              'username that are not either firstname lastname or fullname'],
        #                             content)
        # # except:
        # #     piis = []
        if re.search(r"\nNew Arrivals\nDear\n(\w+)\n                                          (\w+)\n", content):
            res = re.search(r"\nNew Arrivals\nDear\n(\w+)\n                                          (\w+)\n", content)
            piis = [{res.group(1): 'FistName',
                     res.group(2): 'LastName'}]
            logger.success(piis)

        else:
            piis = None
        data['name_only'] = piis
        out.append(data)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_again_extraction(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_again_extraction.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    ins = PIIExtraction()
    out = []
    for data in tqdm.tqdm(datas):
        content = data['sample_content']
        piis = ins.main('ecommerce_v0', content, 3)
        data['again_extracted'] = piis
        out.append(data)
        with open(new_delivery_summary_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def delivery_merge_results(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_again_extraction.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    out = []
    for data in datas:
        prev_piis = data.get('piis')
        full_address_piis = data.get('full_address_piis')
        new_data = {}
        new_piis = {}
        for ad in full_address_piis:
            new_piis[ad['pii_content']] = ad['pii_class']
        to_remove = []
        for k in prev_piis:
            for ad in full_address_piis:
                if k in ad['pii_content'] and "Street" in prev_piis[k]:
                    logger.warning(f"{k} removed.")
                    to_remove.append(k)
                    continue
                if ' ' in k and prev_piis[k] == 'ZipCode':
                    logger.warning(f"{k} removed.")
                    to_remove.append(k)
                    continue
        new_piis.update(prev_piis)
        for i in to_remove:
            if i in new_piis:
                del new_piis[i]
        new_data["piis"] = new_piis
        new_data['sample_content'] = data['sample_content']
        out.append(new_data)
    with open(new_delivery_summary_path, 'w') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)


def TASK_regenerate_templates(template_base: Path):
    templates = glob.glob(str(template_base / '*.txt'))
    templates = [Path(i) for i in templates]
    generate_selected_templates(templates, False)


def delivery_final_sub(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_final_sub.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    new_datas = []
    sub_map = {}
    for num in ['123', '1234', '12345', '123456', '1234567', '12345678', '123456789', '1234567890'][::-1]:
        sub_map[num] = str(random.randint(10 ** (len(num) - 1), 10 ** len(num) - 1))
    for data in datas:
        new_piis = {}
        content = data['sample_content']
        for k in data['piis']:
            if k not in content:
                logger.debug(f"{k} filtered")
                continue
            if 'Street' in data['piis'][k]:
                new_piis[k] = 'Address'
                continue
            if data['piis'][k] == 'PhoneNumber' and re.findall('\d', k):
                if len(re.findall('\d', k)) > 13 or len(re.findall('\d', k)) < 8:
                    g = PersonGenerator()
                    new_piis[g.PhoneNumber] = 'PhoneNumber'
                    content = content.replace(k, g.PhoneNumber)
                    logger.debug(f"{k} =>{g.PhoneNumber}")

                if '123456' in k:
                    g = PersonGenerator()
                    new_piis[g.PhoneNumber] = 'PhoneNumber'
                    content = content.replace(k, g.PhoneNumber)
                    logger.debug(f"{k} =>{g.PhoneNumber}")
                continue
            new_k = k
            for j in sub_map:
                if j in new_k:
                    g = PersonGenerator()
                    if g.__getattribute__(data['piis'][k]):
                        new_k = str(g.__getattribute__(data['piis'][k]))
                        new_piis[str(g.__getattribute__(data['piis'][k]))] = data['piis'][k]
                        content = content.replace(k, str(g.__getattribute__(data['piis'][k])))
                        logger.debug(f"{k} =>{str(g.__getattribute__(data['piis'][k]))}")
                    else:
                        new_k = sub_map[j]
                        new_piis[sub_map[j]] = data['piis'][k]
                        content = content.replace(k, sub_map[j])
                        logger.debug(f"{k} =>{sub_map[j]}")
                    continue

            new_piis[k] = data['piis'][k]
        data["piis"] = new_piis

        for k in sub_map:
            if k in content:
                logger.warning(f"FINAL CONTENT: {k}=> {sub_map[k]}")
                content = content.replace(k, sub_map[k])

        for k in data['piis']:
            if k not in content:
                logger.warning(f"{k} not in {content[:20]}")

        data['sample_content'] = content
        new_datas.append(data)
    with open(new_delivery_summary_path, 'w') as f:
        json.dump(new_datas, f, indent=4, ensure_ascii=False)


def final_formatting(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_final_format.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    new_datas = []
    for data in datas:
        new_data = {'sample_content': data['sample_content'],
                    'piis': data['piis']}
        name_only = data.get('name_only')
        if not name_only:
            new_datas.append(new_data)
            continue
        for d in name_only:
            if 'pii_content' in d:
                new_data['piis'].update({d['pii_content']: d['pii_class']})
            else:
                new_data['piis'].update(d)
    with open(new_delivery_summary_path, 'w') as f:
        json.dump(new_datas, f, indent=4, ensure_ascii=False)

def manual_check(delivery_summary_path: Path):
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    from openpyxl import Workbook
    import pandas
    # 创建一个新的工作簿和工作表
    wb = Workbook()
    ws = wb.active

    out_list = []
    for data in datas:
        out_list.append(data['sample_content'])
    for item in out_list:
        ws.append([item])
    # 保存工作簿
    excel_path = 'out.xlsx'
    wb.save(excel_path)


def filter_by_group(delivery_summary_path: Path):
    new_delivery_summary_path = delivery_summary_path.parent / str(
        delivery_summary_path.stem + "_by_group.json")
    with open(delivery_summary_path, 'r') as f:
        datas = json.load(f)
    new_datas = []
    existing_mark = []
    for data in tqdm.tqdm(datas):
        if 'RECEIPT' in data['sample_pii_extracted_path']:
            mark = re.search('(RECEIPT.*\.png_)', data['sample_pii_extracted_path']).group(1)
            if mark not in existing_mark:
                new_datas.append(data)
                existing_mark.append(mark)
            else:
                continue
        else:
            new_datas.append(data)
    logger.success(f"{len(new_datas)} samples")
    with open(new_delivery_summary_path, 'w') as f:
        json.dump(new_datas, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # batch_paths = get_high_score_templates_batch_paths()
    # res = relocate_templates(list(batch_paths), GOLD_TEMPLATES_HOME)

    # res = get_low_score_templates()
    # print(res)

    # res = get_high_score_samples()
    # deliver_selected_samples(res)

    # selected_templates_base = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/good_template_2')
    # TASK_regenerate_templates(selected_templates_base)

    delivery_shuffle_number(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary.json'))

    # filter_by_group(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary_1741764262.json'))


    # delivery_openai_extraction(Path(
    #     '/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541.json'),
    #                            multiprocess=True)

    # delivery_sample_expand(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741679765/delivery_summary_1741679826.json'))

    # delivery_check_absolute_items(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted.json'))
    # delivery_check_high_freq_items(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741700760/delivery_summary_1741701042_openai35_extracted_absolute.json'))
    # delivery_sub_stupid_stuff(Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary_1741764262_by_group.json"))
    # delivery_only_extract_address(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted_absolute_subbed.json'))
    # delivery_again_extraction(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted_absolute_subbed.json'))
    # delivery_only_full_address(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted_absolute_subbed.json'))
    # delivery_merge_results(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted_absolute_subbed_add_FULL_addressONLY.json'))
    # delivery_final_sub(Path(
    #     '/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary_1741764262_by_group_subbed.json'))
    # delivery_only_names(Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary_1741764262_by_group_subbed_final_sub.json"))
    # final_formatting(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741717212/delivery_summary_1741717541_openai35_extracted_absolute_subbed_add_FULL_addressONLY_again_extraction_final_sub_add_address.json'))
    # manual_check(Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/gold_delivery_home/delivery_1741762895/delivery_summary_1741764262_by_group_subbed_final_sub.json'))