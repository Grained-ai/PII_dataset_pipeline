import random

import glob

import json
from pathlib import Path

path = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/internet_logs")
files = glob.glob(str(path/'*.txt'))
all_lines = []
for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
        all_lines.extend(lines)

random.shuffle(all_lines)
with open('../../../../Desktop/GrainedAI/Datasets/PII/FinalDelivery/InternetLogs/raw_data/InternetLogs.json', 'w') as f:
    json.dump(all_lines[:25001], f, indent=2, ensure_ascii=False)