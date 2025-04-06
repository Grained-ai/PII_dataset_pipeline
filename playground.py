from glob import glob
import PyPDF2

import tqdm
import shutil
from pathlib import Path

def relocate_1():
    images = glob('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/modules/hf_storage/**/images/**/*.png')

    target_folder = '/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce'

    for im in tqdm.tqdm(images):
        im_path = Path(im)
        new_name = '_'.join(['RECEIPT', im_path.parts[-4], im_path.parts[-1]])
        new_path = Path(target_folder)/new_name
        shutil.copy(im_path, new_path)

def relocate_2():
    images = glob('/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/pdfpro/storage/*.png')
    target_folder = '/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/E-commerce'
    for im in tqdm.tqdm(images):
        im_path = Path(im)
        new_name = '_'.join(['ConfirmationPDF', im_path.parts[-1]])
        new_path = Path(target_folder)/new_name
        shutil.copy(im_path, new_path)

def pdf_to_placeholder_template():
    file_path = Path('/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/Legal/ds10.pdf')

    # 打开PDF文件
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # 获取表单字段
        fields = reader.get_fields()

        if not fields:
            print("未找到任何表单字段。")
            return

        # 输出每个字段的名字
        for field_name, field in fields.items():
            print(f"字段名: {field_name}")
            # 可选：打印更多关于字段的信息
            print(f"  类型: {field['/FT']}, 值: {field.get('/V', '无')}\n")
            print(field)


if __name__ == "__main__":
    # relocate_2()
    pdf_to_placeholder_template()