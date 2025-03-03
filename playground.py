from glob import glob

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



if __name__ == "__main__":
    relocate_2()