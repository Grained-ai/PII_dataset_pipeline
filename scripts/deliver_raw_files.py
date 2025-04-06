import hashlib

import random
from pdf2image import convert_from_path

import json
import PyPDF2
from bs4 import BeautifulSoup
import re

import traceback

import shutil

import os
from modules.ocr_handler import OCRHandler
import time
import tqdm
from loguru import logger
from pathlib import Path
from glob import glob
from modules.sample_generation import SampleGeneration
import fitz  # PyMuPDF
from PIL import Image

# PATH_BASE = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds10')
# PATH_BASE = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds7666')
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds6563")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds7781")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds7795")
# PATH_BASE = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds7795')
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds5108")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds5112")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds5154")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds5155")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds5111_h")
# PATH_BASE = Path("/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/ds7677")

# pdfs = glob(str(PATH_BASE/"**"/"*.pdf"))
#
# delivery_path = PATH_BASE/f'delivery_{str(int(time.time()*1000))}'
# os.makedirs(delivery_path, exist_ok=True)
#
# for pdf in pdfs:
#     pdf_path = Path(pdf)
#     target_path = delivery_path/f"{pdf_path.parts[-2]}_{pdf_path.name}"
#     shutil.copy(pdf_path, target_path)

FINALIZED_SAMPLES_BASE = Path(
    '/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/legal_samples/legal_samples_20250325')
SOURCE_DOC_BASE = Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal')
JOB_PACKAGE_PATH = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal/job_package")


def generate_execution_job_package():
    final_pdf_analysis_jsons = glob(str(FINALIZED_SAMPLES_BASE / '**' / "*_analysis.json"))
    cur_job_package_path = JOB_PACKAGE_PATH / f"JOB_{str(int(time.time() * 1000))}"
    source_file_path = cur_job_package_path / 'src'
    os.makedirs(source_file_path, exist_ok=True)
    for pdf_analysis_json in tqdm.tqdm(final_pdf_analysis_jsons):
        doc_name = Path(pdf_analysis_json).name.split("_pdf_analysis.json")[0]
        doc_folder_path = source_file_path / Path(pdf_analysis_json).parts[-2]
        os.makedirs(doc_folder_path, exist_ok=True)
        pdf_path = SOURCE_DOC_BASE / str(doc_name + '.pdf')
        if not pdf_path.exists():
            logger.error(f"{pdf_path} not found.")
            continue

        shutil.copy(pdf_path, doc_folder_path / pdf_path.name)
        shutil.copy(pdf_analysis_json, doc_folder_path / str(Path(pdf_analysis_json).name))
        logger.success(f"JOB: {doc_name} created")


def generate_samples(job_package_path: Path, sample_count_total=3000):
    job_pdf_analysis_files = glob(str(job_package_path / 'src' / '**' / '*_pdf_analysis.json'))
    generated_samples_path = job_package_path / "generated_samples"
    os.makedirs(generated_samples_path, exist_ok=True)
    each_sample_count = sample_count_total // len(job_pdf_analysis_files) + 1
    for job_pdf_analysis_file in tqdm.tqdm(job_pdf_analysis_files):
        logger.info(f"Working on {Path(job_pdf_analysis_file).parts[-2]}")
        pdf_path_candidates = glob(str(Path(job_pdf_analysis_file).parent / "*.pdf"))
        if not pdf_path_candidates:
            raise Exception(f"Cannot find pdf for {job_pdf_analysis_file}")
        pdf_path = pdf_path_candidates[0]
        ins = SampleGeneration()
        logger.info(f"Will generate {each_sample_count} samples for {Path(job_pdf_analysis_file).parts[-2]}")
        current_sample_count = len(
            glob(str(generated_samples_path / Path(job_pdf_analysis_file).parts[-2] / "**" / "*.pdf")))
        logger.info(f"Already Have {current_sample_count} samples")
        for i in range(each_sample_count - current_sample_count):
            try:
                ins.generate_pdf_samples(Path(pdf_path),
                                         batch_base=generated_samples_path,
                                         pdf_analysis_path=Path(job_pdf_analysis_file))
                logger.success(f"Finished sample: {current_sample_count + i}")
            except:
                logger.error(f"Failed sample: {current_sample_count + i}")
        logger.success(f"Finished {Path(job_pdf_analysis_file).parts[-2]}")


def deliver_samples(job_package_path: Path):
    generated_samples = glob(str(job_package_path / "generated_samples" / "**" / "**" / "*.pdf"))
    delivery_path = job_package_path / 'delivery_samples'
    os.makedirs(delivery_path, exist_ok=True)
    for sample in tqdm.tqdm(generated_samples):
        shutil.copy(sample, delivery_path / "_".join(Path(sample).parts[-2:]))


def pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    将PDF文件的每一页转换为图片并保存。

    :param pdf_path: PDF文件路径
    :param output_folder: 输出图片保存的文件夹路径
    :param dpi: 图片分辨率，默认300 DPI
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    images = []
    for page_num in range(total_pages):
        output_filename = os.path.join(output_folder, f"page_{page_num + 1}.png")
        if Path(output_filename).exists():
            logger.success(f"Conversion {pdf_document} page{page_num} finished.")
            images.append(output_filename)
            continue
        page = pdf_document.load_page(page_num)  # 加载页面
        pix = page.get_pixmap(dpi=dpi)  # 渲染页面为图像
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 构建输出文件名
        img.save(output_filename, "PNG")  # 保存图像文件
        print(f"Page {page_num + 1} saved as {output_filename}")
        images.append(output_filename)
    return images


def extract_pdf_content(file_path: Path, ocr_ins, outpath: Path, ocr_only=True):
    ocr_base = Path(outpath) / 'ocr_base'
    flatten_base = Path(outpath) / 'flatten_file'
    os.makedirs(flatten_base, exist_ok=True)
    os.makedirs(ocr_base, exist_ok=True)
    if not ocr_only:
        try:
            with fitz.open(file_path) as doc:
                widgets = []
                for page in doc:
                    # 遍历页面中的所有小部件（如表单字段）
                    for widget in page.widgets():
                        widgets.append(widget)
            if widgets:
                logger.info("Will do flatten First.")
                flatten_file_path = flatten_pdf(input_pdf_path=file_path,
                                                output_pdf_path=flatten_base / "_".join(file_path.parts[-2:]))
                logger.success(f"Flatten to {flatten_file_path}")
                with fitz.open(flatten_file_path) as doc:
                    content = ""

                    for page in doc:
                        content += page.get_text("text") + "\n"
            else:
                with fitz.open(file_path) as doc:
                    content = ""
                    logger.info("[PDF]: Try extract with fitz first")

                    for page in doc:
                        content += page.get_text("text") + "\n"
        except Exception as e:
            # exit(0)
            logger.warning(f"[PDF]: Will use ocr to handle. {e}")
            cur_ocr_base = ocr_base / "_".join(file_path.parts[-2:])
            os.makedirs(cur_ocr_base, exist_ok=True)
            images = pdf_to_images(pdf_path=file_path, output_folder=cur_ocr_base)
            content = ""
            for image in images:
                try:
                    lines = ocr_ins.get_ocr_result_by_block(image, output_path=cur_ocr_base, if_debug=False)
                except:
                    with open(Path(image).parent / "FAIL.txt", 'w') as f:
                        f.write(str(traceback.print_exc()))
                    continue
                sample_content = '\n'.join(lines)
                content += sample_content
    else:
        logger.warning(f"[PDF]: Will use ocr to handle. {file_path}")
        with fitz.open(file_path) as doc:
            widgets = []
            for page in doc:
                # 遍历页面中的所有小部件（如表单字段）
                for widget in page.widgets():
                    widgets.append(widget)
        if widgets:
            logger.info("Will do flatten First.")
            flatten_file_path = flatten_pdf(input_pdf_path=file_path,
                                            output_pdf_path=flatten_base / "_".join(file_path.parts[-2:]))
            logger.success(f"Flatten to {flatten_file_path}")
            file_path = flatten_file_path
        logger.info(f"[PDF]: Now use ocr to handle. {file_path}")

        cur_ocr_base = ocr_base / "_".join(file_path.parts[-2:])
        os.makedirs(cur_ocr_base, exist_ok=True)
        images = pdf_to_images(pdf_path=file_path, output_folder=cur_ocr_base)
        content = ""
        for image in images:
            try:
                lines = ocr_ins.get_ocr_result_by_block(image, output_path=cur_ocr_base, if_debug=False)
            except Exception as e:
                with open(Path(image).parent / "FAIL.txt", 'w') as f:
                    f.write(str(traceback.print_exc()))
                continue
            sample_content = '\n'.join(lines)
            content += sample_content
    return content


def extract_html_content(file_path: Path):
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as f:
        sample_text = f.read()
    soup = BeautifulSoup(sample_text, 'lxml')
    texts = [element.get_text(strip=True) for element in soup.find_all(text=True) if
             element.get_text(strip=True)]
    text_all = '\n'.join(texts)
    text_all = re.sub(r"\n+", "\n", text_all)
    return text_all


def deliver_raw_text_files(base_path: Path, output_path: Path):
    ins = OCRHandler()
    files = [i for i in base_path.rglob("*.*") if i.is_file()]
    files = [i for i in files if
             "part" not in i.name and "ocr_result" not in i.name and 'content.txt' not in i.name and i.name != ".DS_Store"]
    ocr_base = output_path / f'ocr_base'
    os.makedirs(ocr_base, exist_ok=True)
    for file in tqdm.tqdm(files):
        file = Path(file)
        content_output_path = file.parent / (file.stem + "_content.txt")
        # if content_output_path.exists():
        #     logger.success("already finished")
        #     continue
        try:
            if file.suffix in ['.pdf']:
                content = extract_pdf_content(file, ins, output_path, ocr_only=True)

            elif file.suffix in ['.png', '.jpg']:
                logger.warning(f"Image: use ocr to handle {file}")
                cur_ocr_base = ocr_base / "_".join(file.parts[-2:])
                os.makedirs(cur_ocr_base, exist_ok=True)
                lines = ins.get_ocr_result_by_block(file, output_path=cur_ocr_base, if_debug=False)
                content = '\n'.join(lines)
            elif file.suffix in ['.html']:
                logger.warning(f"Image: use BS4 to handle {file}")
                content = extract_html_content(file)

            else:
                logger.error(f"{file} failed.")
                continue
        except Exception as e:
            logger.error(f"{file} failed to extract plaintext: {e}")
            continue
        with open(content_output_path, 'w') as f:
            f.write(content)


def pdf_to_images_2(pdf_path, output_folder, if_ignore_no_placeholder_pages=False):
    """
    将给定的 PDF 文件转换为图像，并保存到指定文件夹。

    :param pdf_path: PDF 文件的路径。
    :param output_folder: 保存图像的文件夹路径。
    :param if_ignore_no_placeholder_pages: 如果为 True，则仅转换包含占位符的页面，默认为 False。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将 PDF 转换为图像
    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        # 构建输出图像的文件名
        image_filename = os.path.join(output_folder, f'page_{i + 1}.png')
        # 保存图像
        image.save(image_filename, 'PNG')

    print(f"PDF 已转换为图像并保存在文件夹: {output_folder}")


def flatten_pdf(input_pdf_path, output_pdf_path):
    """
    将填写后的 PDF 表单平坦化，使其不可编辑，包括处理文本字段、复选框和组合框。

    :param input_pdf_path: 输入的已填写 PDF 表单的路径。
    :param output_pdf_path: 输出的平坦化 PDF 文件的路径。
    """
    # 打开 PDF 文档
    logger.info(f"Open {input_pdf_path}")
    with fitz.open(input_pdf_path) as doc:

        # 遍历每一页
        for page in doc:
            # 遍历页面中的所有小部件（如表单字段）
            for widget in page.widgets():
                rect = widget.rect
                field_type = widget.field_type
                field_value = widget.field_value

                # 获取字体名称和大小，设置默认值
                fontname = widget.__dict__.get("font_name", "helv")
                fontsize = widget.__dict__.get("text_fontsize", 12)
                widget_button_state = widget.button_states()

                if field_type == 2 and not widget_button_state:
                    # 文本字段
                    if field_value:
                        # 计算文本插入位置
                        insert_x = rect.x0 + 2  # 左边距偏移
                        insert_y = rect.y1 - 2  # 底部对齐偏移
                        page.insert_text((insert_x, insert_y), field_value, fontsize=fontsize, fontname=fontname)
                elif field_type == 4 or widget_button_state:  # 复选框
                    # 获取复选框的可能状态
                    states = widget.button_states()
                    if not states:
                        continue
                    on_states = states.get('normal', []) + states.get('down', [])
                    on_states = [state for state in on_states if state.lower() != 'off']

                    if field_value in on_states:  # 复选框被选中
                        # 计算“X”符号的插入位置
                        insert_x = rect.x0 + rect.width / 4
                        insert_y = rect.y0 + fontsize
                        page.insert_text((insert_x, insert_y), "X", fontsize=rect.height, fontname="helv")
                elif field_type == 7:  # 组合框
                    if field_value:
                        # 计算文本插入位置
                        insert_x = rect.x0 + 2
                        insert_y = rect.y1 - 2
                        page.insert_text((insert_x, insert_y), field_value, fontsize=fontsize, fontname=fontname)
                # 删除小部件
                page.delete_widget(widget)

        # 保存平坦化后的 PDF
        doc.save(output_pdf_path)
    return output_pdf_path


def convert_pdfs(delivery_path: Path):
    pdfs = glob(str(delivery_path / "*.pdf"))
    for pdf in tqdm.tqdm(pdfs):
        pdf = Path(pdf)

        flatten_pdf(pdf, pdf.parent / str(pdf.name.split("_")[0] + ".pdf"))


def flatten_all(folder: Path):
    pdfs = glob(str(folder / "*.pdf"))
    pdfs = [i for i in pdfs if 'flatten' not in i]
    random.shuffle(pdfs)
    for pdf in tqdm.tqdm(pdfs):
        pdf = Path(pdf)
        flatten_pdf(pdf, pdf.parent / str(pdf.stem + '_flatten.pdf'))


def extract_text_from_pdf(pdf_path):
    """
    从指定的 PDF 文件中提取文本。

    :param pdf_path: PDF 文件的路径。
    :return: 提取的文本内容。
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text


def process_hi_data():
    hi_main = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/HiData_samples")
    part_0_original = hi_main / ''


def process_g_legal():
    f_files_base = Path("/Legal_all/legal_pdfs/flatten_file")
    pdf_ocr_source = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal_all/legal_pdfs/legal_docs")
    target_path = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/FinalDelivery/Legal/sources/g_legal")
    for f_path in glob(str(f_files_base / '*.pdf')):
        f_path = Path(f_path)
        target_pdf_path = pdf_ocr_source / f_path.name.split('_')[-1]
        target_ocr_res_path = target_pdf_path.parent / str(target_pdf_path.stem + "_content.txt")
        shutil.copy(target_pdf_path, target_path / target_pdf_path.name)
        shutil.copy(target_ocr_res_path, target_path / target_ocr_res_path.name)

def hash_filename(file_path: Path) -> str:
    """对文件名（不含扩展名）进行哈希，并保留原扩展名"""
    base_name = '_'.join(file_path.parts[-2:])  # 获取文件名（不含扩展名）
    ext = file_path.suffix  # 获取扩展名
    hashed_name = hashlib.md5(base_name.encode()).hexdigest()+str(int(time.time()*1000))
    return hashed_name+ext

def deliver_final_delivery_folder():
    final_delivery_path = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/FinalDelivery")
    sub_folders = glob(str(final_delivery_path / "*"))
    sub_folders = [i for i in sub_folders if '.json' not in i]
    all_summary = {}
    for sub_folder in tqdm.tqdm(sub_folders):
        sub_folder = Path(sub_folder)
        logger.info(f"Working on {sub_folder.name}")
        sources_path = sub_folder / 'sources'

        todo_files = (list(sources_path.rglob("*.jpg")) +
                      list(sources_path.rglob("*.png")) +
                      list(sources_path.rglob("*.html")) +
                      list(sources_path.rglob("*.pdf")))
        todo_files = list(set([f for f in todo_files if f.is_file() and f.name != ".DS_Store" and 'part' not in f.name]))
        ready_files = []
        not_ready_files = []
        for todo_file in todo_files:
            todo_file = Path(todo_file)
            if 'part' not in todo_file.name:
                content_file = todo_file.parent/str(todo_file.stem+"_content.txt")
            else:
                content_file = todo_file.parent/str(todo_file.stem+"_content.txt").replace('_part_0', '')
            if content_file.exists():
                ready_files.append(todo_file)
            else:
                not_ready_files.append(todo_file)
        logger.info(f"READY: {len(ready_files)}")
        logger.info(f"Not READY: {len(not_ready_files)}")
        raw_files_path = sub_folder/'raw_data'/'documents'
        shutil.rmtree(raw_files_path)
        os.makedirs(raw_files_path, exist_ok=False)
        summary = {}
        for todo_file in ready_files:
            if 'part' not in todo_file.name:
                content_file = todo_file.parent / str(todo_file.stem + "_content.txt")
            else:
                content_file = todo_file.parent / str(todo_file.stem + "_content.txt").replace('_part_0', '')
            new_file_name = hash_filename(todo_file)
            shutil.copy(todo_file, raw_files_path/new_file_name)
            with open(content_file, 'r') as f:
                content = f.read()
            summary[str(raw_files_path/new_file_name)] = {"original": str(todo_file), 'content': content}
        raw_data_json_path = raw_files_path.parent/str(sub_folder.name+".json")
        raw_data_json_content = list(set([i['content'] for i in summary.values()]))
        if raw_data_json_content:
            with open(raw_data_json_path, 'w') as f:
                json.dump(raw_data_json_content, f, indent=4, ensure_ascii=False)

            logger.success(f"RAW_DATA_JSON: {raw_data_json_path}")
        else:
            logger.error(f"RAW_DATA is empty: {raw_data_json_path}")
        all_summary[sub_folder.name] = summary
    with open(final_delivery_path/'all_summary.json', 'w') as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=4)

def analyze_summary():
    summary= Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/FinalDelivery/all_summary.json")
    with open(summary, 'r') as f:
        data = json.load(f)

    for k in data.keys():
        all_data_path = summary.parent/k/'raw_data'/f"{k}.json"
        with open(all_data_path, 'r') as f:
            d = json.load(f)

            print(k, len(d))




if __name__ == "__main__":
    # generate_execution_job_package()
    # generate_samples(job_package_path=Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal/job_package/JOB_1742960832638'),
    #                  sample_count_total=3000)
    # deliver_samples(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal/job_package/JOB_1742960832638'))
    # deliver_raw_text_files(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Financial-samples"),
    #                        Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Financial-samples/temp_path"))
    # deliver_raw_text_files(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce-samples"),
    #                        Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce-samples/temp_path"))
    # summarize_raw_text(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Financial-samples/temp_path"))
    # summarize_raw_text(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal-samples/temp_path"))
    # convert_pdfs(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal-samples/raw_files/delivery_samples"))
    # print(pdf_to_images_2(Path('/Users/anthonyf/Desktop/1742984553880.pdf'), '.'))
    # flatten_pdf(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/legal_pdfs/legal_docs/1742984966617.pdf'), Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/legal_pdfs/legal_docs/1742984966617_flatten.pdf'))
    # flatten_all(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/legal_pdfs/legal_docs'))

    # path = "/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal-samples/legal_sample_3526.json"
    # with open(path, 'r') as f:
    #     data = json.load(f)
    # data = list(set(data))
    # print(len(data))
    # print(extract_text_from_pdf(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/legal_pdfs/legal_docs/1742962335983_flatten.pdf")))

    # deliver_raw_text_files(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal_all/legal_pdfs'),
    #                        output_path=Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Official/Legal/temp_path'))
    # deliver_raw_text_files(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/TagX-samples'),
    #                        output_path=Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Official/TagX/temp_path'))
    # deliver_raw_text_files(Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/HiData_samples'),
    #                            output_path=Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Official/HiData/temp_path'))
    # deliver_final_delivery_folder()
    deliver_raw_text_files(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/FinalDelivery/Financial/sources/hi_Financial/insurance_claims_2"),
                           output_path=Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Official/insurance_claims"))
    # deliver_raw_text_files(Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/FinalDelivery/Financial/sources"),
    #                        output_path=Path(
    #                            "/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Official/financial_sources_final"))
    # analyze_summary()