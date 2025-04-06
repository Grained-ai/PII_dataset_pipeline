import glob
import json
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import tqdm
from loguru import logger

from langchain_community.callbacks import get_openai_callback
from modules.PII_extraction import PIIExtraction
from modules.sample_generation import SampleGeneration
from modules.sample_scoring import SampleScoring
from multiprocessing import current_process, Pool, Lock
import undetected_chromedriver as uc

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

DEFAULT_CHROMEDRIVER_PATH = '/Users/anthonyf/Desktop/Tools/chromedriver/chromedriver'
DEFAULT_CHROMEDRIVER_VERSION = 133


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class PIISampleGenPipeline:
    def __init__(self):
        self.pii_extractor = PIIExtraction()
        self.sample_generator = SampleGeneration()
        self.sample_scoring_ins = SampleScoring()

    def convert_paths_to_strings(self, d):
        if isinstance(d, dict):
            return {k: self.convert_paths_to_strings(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.convert_paths_to_strings(element) for element in d]
        elif isinstance(d, Path):
            return str(d)
        else:
            return d

    @staticmethod
    def html_to_image(html_path: Path, output_path):
        logger.info(f"Try to html2image: {html_path} to {output_path}")
        if Path(output_path).exists() or (output_path.parent / (f'sample_{Path(output_path).stem}.png')).exists():
            logger.success(f'{html_path} already to png')
            return output_path.parent / (f'sample_{Path(output_path).stem}.png')
        logger.info(f"Still need to convert {output_path}")
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
        chrome_options.page_load_strategy = 'eager'

        driver = uc.Chrome(options=chrome_options,
                           driver_executable_path=DEFAULT_CHROMEDRIVER_PATH,
                           version_main=DEFAULT_CHROMEDRIVER_VERSION)

        driver.get(f"file://{html_path.absolute()}")

        # 等待直到文档加载完成
        WebDriverWait(driver, 120).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )

        # 获取整个页面的高度和宽度
        height = driver.execute_script("return document.body.scrollHeight")
        width = driver.execute_script("return document.body.scrollWidth")

        # 动态设置窗口大小以匹配内容尺寸
        driver.set_window_size(width, height)

        # 对body元素进行截图
        body = driver.find_element(By.TAG_NAME, 'body')
        body.screenshot(str(output_path))

        # 关闭浏览器
        driver.quit()
        if output_path.exists():
            return output_path
        else:
            time.sleep(5)
            return output_path

    @staticmethod
    def unit_process_input_body(task):
        if not task:
            return
        template, batch_path, template_only, sample_count = task
        logger.info(f"Process {current_process().name} starts to work on {template}")
        with get_openai_callback() as cb:
            sample_generator = SampleGeneration()
            try:  # 假设SampleGenerator已正确定义和初始化
                batch_details = sample_generator.main('ecommerce', Path(template), batch_dir=batch_path,
                                                      sample_count=sample_count, template_only=template_only)
                batch_details = {} if batch_details is None else batch_details
            except:
                batch_details = {}
                logger.error(traceback.format_exc())
                logger.error(task)
        batch_details['tokens'] = cb.total_tokens
        with open(batch_path / f'batch_status_{str(int(time.time()))}.json', 'w') as f:
            json.dump(batch_details, f, indent=2, ensure_ascii=False)
        logger.success(json.dumps(batch_details, ensure_ascii=False, indent=4, cls=PathEncoder))
        # self.check_current_out_count()

    @staticmethod
    def unit_batch_sample_generation(task):
        batch_path, sample_count = task
        with get_openai_callback() as cb:
            sample_generator = SampleGeneration()
            try:  # 假设SampleGenerator已正确定义和初始化
                batch_details = sample_generator.retry_batch(batch_path,
                                                             sample_count=sample_count,
                                                             template_only=False)
                batch_details = {} if batch_details is None else batch_details
            except:
                batch_details = {}
                logger.error(traceback.format_exc())
                logger.error(batch_path)
        batch_details['tokens'] = cb.total_tokens
        with open(batch_path / f'batch_status_{str(int(time.time()))}.json', 'w') as f:
            json.dump(batch_details, f, indent=2, ensure_ascii=False)
        logger.success(json.dumps(batch_details, ensure_ascii=False, indent=4, cls=PathEncoder))
        return batch_details

    def unit_extract_pii_labels(self, pii_category, pii_category_labels, sample_content_path: Path, delivery_base: Path,
                                placeholder_content_map=None,
                                sample_doc=None, person_mapping=None, sample_score=None, score_reason=None):
        out = {}
        piis = []
        with open(sample_content_path, 'r') as f:
            sample_content = f.read()
        out['sample_content'] = sample_content
        out['sample_score'] = sample_score,
        out['score_reason'] = score_reason
        extracted_piis_all = []
        extraction_out_path = sample_content_path.parent/str(sample_content_path.stem+"_pii_extraction.json")
        if sample_score and sample_score >= 7:
            if extraction_out_path.exists():
                with open(extraction_out_path, 'r') as f:
                    extracted_piis = json.load(f)
                    logger.warning("Already extracted PIIs")
            else:
                logger.success("High score sample, do extraction.")
                ins = PIIExtraction()
                extracted_piis = ins.main(pii_category=pii_category,
                                          input_str=sample_content,
                                          votes=3)
                with open(extraction_out_path, 'w') as f:
                    json.dump(extracted_piis, f, ensure_ascii=False, indent=2)
                    logger.warning(f"PIIS stored at {extraction_out_path}")
                del ins
            for r in extracted_piis:
                extracted_piis_all.append({r['pii_content']: r['pii_class']})
            out['extracted_piis'] = extracted_piis_all

        if not placeholder_content_map:
            out['pii_extraction_method'] = 'extraction'
            # res = self.pii_extractor.extract(pii_category=pii_category,
            #                                  input_str=sample_content,
            #                                  votes=11)
            # for r in res:
            #     piis.append({r['pii_content']: r['pii_class']})
            # out['piis'] = piis

        else:
            with open(placeholder_content_map, 'r') as f:
                placeholder_content_map_data = json.load(f)
            if not placeholder_content_map_data:
                out['pii_extraction_method'] = 'extraction'
                # res = self.pii_extractor.extract(pii_category=pii_category,
                #                                  input_str=sample_content,
                #                                  votes=11)
                # for r in res:
                #     piis.append({r['pii_content']: r['pii_class']})
                # out['pii_extraction_method'] = 'extraction'
                # out['piis'] = piis
            else:
                p_c_mapping = {}
                for p_c in placeholder_content_map_data:
                    try:
                        p_c_mapping.update(p_c)
                    except:
                        print("HERE")
                for p_name in pii_category_labels:
                    for placeholder in p_c_mapping:
                        if p_name in placeholder:
                            piis.append({p_c_mapping[placeholder]: p_name})
                            logger.success(f"{p_name} => {placeholder}: {p_c_mapping[placeholder]}")
                # out['pii_extraction_method'] = 'placeholder'
                out['pii_extraction_method'] = 'placeholder'
        out['piis'] = piis

        if sample_doc:
            if Path(sample_doc).suffix in ['.html']:
                out['sample_doc'] = sample_doc
                # try:
                #     image_path = self.html_to_image(Path(sample_doc),
                #                                     delivery_base / str(sample_content_path.stem + ".png"))
                #     sample = image_path
                #     dst_path = delivery_base / f'sample_{sample_content_path.stem}{sample.suffix}'
                #     if sample != dst_path:
                #         shutil.move(sample, dst_path)
                #     out['sample_doc'] = dst_path
                # except Exception as e:
                #     logger.error(e)
                #     logger.error(traceback.print_exc())
                #     out['sample_doc'] = sample_doc
            else:
                sample = Path(sample_doc)
                dst_path = delivery_base / f'sample_{sample_content_path.stem}{sample.suffix}'
                shutil.copy(sample, dst_path)
                out['sample_doc'] = dst_path
        return out

    def unit_packup_file(self, task):
        batch_dir, delivery_path = task
        task_status_files = glob.glob(str(batch_dir / 'batch_status*.json'))
        if not task_status_files:
            return
        samples = []
        pii_category = 'ecommerce'
        for task_status_file in task_status_files:
            try:
                with open(task_status_file, 'r') as f:
                    task_status = json.load(f)
                samples.extend(task_status.get('generated_samples', []))
                pii_category = task_status.get('pii_category') if task_status.get('pii_category') else pii_category
            except Exception as e:
                logger.error(f"STATUS JSON {task_status_file} OPEN FAILED: {traceback.print_exc()}")
                continue
        if not samples:
            return
        batch_out = []
        pii_configs = self.pii_extractor.load_config(pii_category)
        pii_categories = [i['key_name'] for i in pii_configs[0]]
        samples_pmap = list(set([i.get('sample_content') for i in samples if i.get('sample_content')]))
        samples = [i for i in samples if i.get('sample_content') in samples_pmap]
        for sample in samples:
            sample_doc = sample.get('sample_doc') if not sample.get(
                'html_sample') else sample.get('html_sample')
            sample_doc = None if not sample_doc else Path(sample_doc)
            sample_score_path = Path(sample.get('sample_content')).parent / str(
                Path(sample.get('sample_content')).stem + "_score.json")
            if sample_score_path.exists():
                with open(sample_score_path, 'r') as f:
                    sample_score_data = json.load(f)
                score = sample_score_data.get('final_score')
                reason = sample_score_data.get('reason')
            else:
                score = None
                reason = None
            logger.debug(score)
            result = self.unit_extract_pii_labels(pii_category=pii_category,
                                                  pii_category_labels=pii_categories,
                                                  sample_content_path=Path(sample.get('sample_content')),
                                                  delivery_base=delivery_path,
                                                  placeholder_content_map=Path(sample.get('placeholder_content_map')),
                                                  sample_doc=sample_doc,
                                                  person_mapping=Path(sample.get('person_mapping')),
                                                  sample_score=score,
                                                  score_reason=reason)
            batch_out.append(result)
        return batch_out

    def unit_rate(self, sample: dict):
        sample_path = sample['sample_content']
        score_path = Path(sample_path).parent / str(Path(sample_path).stem + "_score.json")
        with open(sample['sample_content'], 'r') as f:
            sample_content = f.read()
        seed_content = sample['seed_content']
        score_instance = SampleScoring()
        try:
            res = score_instance.unit_score_synthetic_sample(seed_content, sample_content)
        except:
            return
        res['seed_content'] = seed_content
        res['generated_content'] = sample_content
        with open(score_path, 'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        logger.success(f"Stored at {score_path}")

    def rate_all_samples(self, output_folder_path: Path, multiprocess=False, skip_exists=False):
        batch_statuses = glob.glob(str(output_folder_path / '*' / "batch_status*"))
        all_samples = []
        for batch_status in batch_statuses:
            with open(batch_status, 'r') as f:
                data = json.load(f)
            seed_content = data.get('seed_content')
            if data.get('generated_samples', []):
                generated_sample = data.get('generated_samples', [])
                for i in generated_sample:
                    i['seed_content'] = seed_content
                all_samples.extend(data.get('generated_samples'))
        samples_pmap = list(set([i.get('sample_content') for i in all_samples if i.get('placeholder_content_map')]))
        samples = [i for i in all_samples if i.get('sample_content') in samples_pmap]
        logger.info(f"{len(samples)} samples in total")
        if skip_exists:
            todo_samples = []
            for sample in samples:
                sample_path = sample['sample_content']
                score_path = Path(sample_path).parent / str(Path(sample_path).stem + "_score.json")
                if score_path.exists():
                    logger.success(f"{sample_path} already scored.")
                    continue
                todo_samples.append(sample)
            logger.warning(f"Will skip finished. {len(samples)} => {len(todo_samples)}")
            samples = todo_samples

        if not multiprocess:
            for sample in tqdm.tqdm(samples):
                self.unit_rate(sample)
        else:
            num_processes = 20  # 根据CPU核心数自动选择进程数
            with Pool(processes=num_processes) as pool:
                # 使用tqdm包装pool.imap以显示进度条
                for _ in tqdm.tqdm(pool.imap_unordered(self.unit_rate, samples), total=len(samples)):
                    pass

    def general_pipeline(self, pii_category, output_folder_path, input_folder_path, multi_process=False,
                         template_only=True, generation_only=False, delivery_only=False, unit_batch_sample_count=5):
        task_path = output_folder_path / pii_category
        delivery_path = task_path / 'delivery'
        os.makedirs(delivery_path, exist_ok=True)

        delivery_summary = delivery_path / 'delivery_summary.json'
        batch_out_total = []
        lock = Lock()

        if delivery_only:
            batch_with_template = glob.glob(str(task_path / '**' / '*template.txt'))
            todo_batches = [Path(i).parent for i in batch_with_template]
            tasks = []
            batch_out_total = []
            for todo_batch in todo_batches:
                tasks.append((todo_batch, delivery_path))

            def log_result(result):
                """Append result to the global list with thread-safe operation."""
                with lock:
                    if result:
                        batch_out_total.extend(result)

            if multi_process:
                with Pool(processes=10) as pool:  # 根据需要调整进程数
                    for _ in tqdm.tqdm(pool.imap_unordered(self.unit_packup_file, tasks), total=len(tasks)):
                        pass  # 这里不需要做额外的事情，只需等待所有任务完成
                with Pool(processes=10) as pool:
                    for result in pool.imap_unordered(self.unit_packup_file, tasks):
                        log_result(result)
                        # 所有任务完成后写入JSON文件
                with open(delivery_summary, 'w') as f:
                    json.dump(self.convert_paths_to_strings(batch_out_total), f, indent=4, ensure_ascii=False)
            else:
                for task in tqdm.tqdm(tasks):
                    batch_out = self.unit_packup_file(task)
                    if batch_out:
                        batch_out_total.extend(batch_out)
                        # 写入JSON文件
                        with open(delivery_summary, 'w') as f:
                            json.dump(self.convert_paths_to_strings(batch_out_total), f, indent=4, ensure_ascii=False)

            return delivery_summary

        if generation_only:
            batch_with_template = glob.glob(str(task_path / '**' / '*template.txt'))
            todo_batches = [Path(i).parent for i in batch_with_template]
            tasks = []
            for todo_batch in todo_batches:
                tasks.append((todo_batch, unit_batch_sample_count))
            if multi_process:
                with Pool(processes=10) as pool:  # 根据需要调整进程数
                    for _ in tqdm.tqdm(pool.imap_unordered(self.unit_batch_sample_generation, tasks), total=len(tasks)):
                        pass  # 这里不需要做额外的事情，只需等待所有任务完成

            else:
                for task in tqdm.tqdm(tasks):
                    self.unit_batch_sample_generation(task)

        else:
            input_bodies = [i for i in glob.glob(str(Path(input_folder_path) / '*.*')) if 'part_' not in i]

            tasks = []
            for input_body in input_bodies[:]:
                batch_path = task_path / Path(input_body).stem
                os.makedirs(batch_path, exist_ok=True)
                tasks.append((input_body, batch_path, template_only, unit_batch_sample_count))
            tasks = [i for i in tasks if i]
            if multi_process:
                with Pool(processes=10) as pool:  # 根据需要调整进程数
                    for _ in tqdm.tqdm(pool.imap_unordered(self.unit_process_input_body, tasks), total=len(tasks)):
                        pass  # 这里不需要做额外的事情，只需等待所有任务完成
            else:
                for task in tqdm.tqdm(tasks):
                    self.unit_process_input_body(task)
            return


if __name__ == "__main__":
    ins = PIISampleGenPipeline()
    # ins.generate_e_commerce_job_1_1(True)
    # ins.general_pipeline(pii_category='ecommerce',
    #                      output_folder_path=Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output_2"),
    #                      input_folder_path=Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce-HTML"),
    #                      multi_process=True,
    #                      generation_only=False,
    #                      delivery_only=False,
    #                      template_only=False,
    #                      unit_batch_sample_count=1)
    htmls = glob.glob(str(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce-samples')/'**'/'*.html'))
    print(len(htmls))
    for html in tqdm.tqdm(htmls):
        try:
            ins.html_to_image(Path(html), Path(html).parent/str(Path(html).stem+".png"))

        except:
            logger.error(f"ERROR:{html}")
            continue

    # ins.rate_all_samples(Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/ecommerce"), True, True)
    # ins.unit_extract_pii_labels(pii_category='ecommerce',
    #                             sample_content_path='')
    # ins.html_to_image(Path(
    #     '/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/job_1_v2/0fa601bf-a6bc-51b8-2d35-eade9cb8ab34/0fa601bf-a6bc-51b8-2d35-eade9cb8ab34_0.html'),
    #     'demo.png')
