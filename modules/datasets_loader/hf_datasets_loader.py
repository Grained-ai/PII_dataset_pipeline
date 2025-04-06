import os

from PIL import Image
from loguru import logger
from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path


class HuggingfaceDatasetsLoader:
    def __init__(self, storage_base=None):
        if not storage_base:
            self.storage_base = Path(__file__).parent.parent / 'hf_storage'
        else:
            self.storage_base = storage_base
        os.makedirs(self.storage_base, exist_ok=True)

    def download_datasets_to_local(self, dataset_path: str, if_return=False, retrieve_image=False):
        parts = dataset_path.split('/')
        user, dataset_name = (parts if len(parts) == 2 else (None, parts[0]))

        save_dir = self.storage_base / dataset_name

        if user is not None:
            dataset = load_dataset(dataset_path)
        else:
            dataset = load_dataset(path=dataset_path, streaming=True)

        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_save_dir = save_dir / split_name
            # Save each split individually
            if isinstance(split_data, Dataset):
                split_data.save_to_disk(str(split_save_dir))
            else:
                raise ValueError(f"Unexpected type {type(split_data)} for split data. Expected Dataset.")
            if retrieve_image:
                logger.info("Toggle retrieve image mode. Will download image content in to separate folder")
                image_save_dir = save_dir / 'images' / split_name
                os.makedirs(image_save_dir, exist_ok=True)
                to_retrieve_columns = []
                for column_name in split_data.column_names:
                    column_example = split_data[column_name][0]
                    if isinstance(column_example, Image.Image):
                        to_retrieve_columns.append(column_name)
                for column_name in to_retrieve_columns:
                    for index, image_data in enumerate(split_data[column_name]):
                        file_path = image_save_dir / f"{str(index)}.png"  # 根据实际情况调整扩展名
                        image_data.save(file_path)  # 保存图像
                logger.success(f"Retrieve image mode on. Stored at: {image_save_dir}")

        logger.success(f"Dataset {dataset_name} has been downloaded and saved to {save_dir}")
        if if_return:
            return dataset

    def load_dataset(self, dataset_path: str):
        parts = dataset_path.split('/')
        user, dataset_name = (parts if len(parts) == 2 else (None, parts[0]))
        save_dir = self.storage_base / dataset_name

        if not save_dir.exists():
            logger.warning(f"Directory {save_dir} does not exist. Downloading dataset...")
            return self.download_datasets_to_local(dataset_path, if_return=True)

        existing_splits = [d.name for d in save_dir.iterdir() if d.is_dir()]

        if existing_splits:
            logger.info(f"Loading dataset {dataset_name} from local storage.")
            splits = {}
            for split in existing_splits:
                split_dir = save_dir / split
                if split_dir.exists():
                    try:
                        splits[split] = Dataset.load_from_disk(str(split_dir))
                    except Exception as e:
                        logger.error(f"Failed to load split {split}: {e}")
                        return self.download_datasets_to_local(dataset_path, if_return=True)
            dataset = DatasetDict(splits)
        else:
            logger.warning(f"No splits found in {save_dir}. Downloading dataset...")
            dataset = self.download_datasets_to_local(dataset_path, if_return=True)

        return dataset


if __name__ == "__main__":
    loader = HuggingfaceDatasetsLoader()
    dataset = loader.download_datasets_to_local(
        'AjitRawat/invoiceReplicated', retrieve_image=True)  # Replace with your actual dataset path on Hugging Face Hub
