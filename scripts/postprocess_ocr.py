import os
import traceback
from pathlib import Path

from glob import glob

import tqdm

folder = Path('/Financial-samples')
# GOLD_SAMPLES_HOME = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/good_samples')
TEMP_SAVE = Path(__file__).parent/'storage'
PROCESS_RESULT = TEMP_SAVE/'process'
RAW_IMAGES = TEMP_SAVE/'financial-samples'
os.makedirs(TEMP_SAVE, exist_ok=True)
pdf_receipts = glob(str(folder/'**'/'*.pdf'))
png_receipts = glob(str(folder/'**'/'*.png'))
from loguru import logger
import fitz  # PyMuPDF
from PIL import Image
import os


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

# print(len(receipts))

# receipts = glob(str(RAW_IMAGES/'**'/'*.png'))
# receipts = [i for i in receipts if 'part_' not in i]
# print(len(receipts))
# # exit(0)

## OCR PDF
from modules.ocr_handler import OCRHandler
ins = OCRHandler()
error_list = []
for file in tqdm.tqdm(pdf_receipts):
    try:
        images = pdf_to_images(file, output_folder=str(RAW_IMAGES/'_'.join(Path(file).parts[-2:])))
    except:
        error_list.append(file)
        continue
    for image in images:
        file_out_path = Path(image).parent/(str(Path(image).stem)+"_content.txt")
        if file_out_path.exists():
            logger.success(f"{image} already exists.")
            continue
        try:
            lines = ins.get_ocr_result_by_block(image, output_path=Path(image).parent)
        except:
            with open(Path(image).parent/"FAIL.txt", 'w') as f:
                f.write(str(traceback.print_exc()))
            continue
        sample_content = '\n'.join(lines)
        with open(file_out_path, 'w') as f:
            f.write(sample_content)
print(error_list)
### OCR PNG
# from loguru import logger
#
# from modules.ocr_handler import OCRHandler
# ins = OCRHandler()
# error_list = []
# for file in tqdm.tqdm(png_receipts):
#     images = [file]
#     for image in images:
#         file_out_path = Path(image).parent/(str(Path(image).stem)+"_content.txt")
#         if file_out_path.exists():
#             logger.success(f"{image} already exists.")
#             continue
#         try:
#             lines = ins.get_ocr_result_by_block(image, output_path=Path(image).parent)
#         except:
#             with open(Path(image).parent/"FAIL.txt", 'w') as f:
#                 f.write(str(traceback.print_exc()))
#             continue
#         sample_content = '\n'.join(lines)
#         with open(file_out_path, 'w') as f:
#             f.write(sample_content)
# print(error_list)