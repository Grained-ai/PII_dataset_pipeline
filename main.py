import glob
import os
from pathlib import Path

import tqdm
from loguru import logger

from langchain_community.callbacks import get_openai_callback
from modules.PII_extraction import PIIExtraction
from modules.sample_generation import SampleGeneration


class PIISampleGenPipeline:
    def __init__(self):
        self.pii_extractor = PIIExtraction()
        self.sample_generator = SampleGeneration()

    def generate_e_commerce_job_0(self):
        """
        Job 0: E-commerce, 120+ templates images
        :return:
        """
        with get_openai_callback() as cb:
            template_images_path = "/Users/anthonyf/projects/grainedAI/dataset_downloader/scripts/pdfpro/storage"
            templates = glob.glob(str(Path(template_images_path)/'*template*.png'))
            for template in tqdm.tqdm(templates[:1]):
                logger.info(f"Starts to work on {template}")
                batch_path = Path(__file__).parent/'output'/'job_0'/Path(template).stem
                os.makedirs(batch_path, exist_ok=True)
                self.sample_generator.main(Path(template), batch_dir=batch_path)
            logger.success(f"Total token usage: {cb.total_tokens}")


if __name__ =="__main__":
    ins = PIISampleGenPipeline()
    ins.generate_e_commerce_job_0()