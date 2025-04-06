import random

import json
import tqdm

from pathlib import Path
from glob import glob

DELIVERY_PATH = Path('/Users/anthonyf/projects/grainedAI/PII_dataset_pipeline/output/social_media_samples')
BASE_PATH = Path("/Users/anthonyf/Desktop/GrainedAI/Datasets/PII/social_media")
delivery_file_path = DELIVERY_PATH/'SocialMedia.json'

# mind_res = glob(str(BASE_PATH/"*_res.json"))
#
# total_res = []
# for hash_file in tqdm.tqdm(mind_res):
#     with open(hash_file, 'r') as f:
#         data = json.load(f)
#     for i in data['outs']:
#         i2 = i.replace("mind", '').replace("more_horiz\n", '')
#         total_res.append(i2)
#
# total_res = list(set(total_res))
#
# random.shuffle(total_res)
#
# with open(delivery_file_path, 'w') as f:
#     json.dump(total_res, f, indent=2, ensure_ascii=False)

##### USERS
#
# mind_person_info = BASE_PATH/'user_abouts.json'
#
# with open(delivery_file_path, 'r') as f:
#     current_data = json.load(f)
#
# with open(mind_person_info, 'r') as f:
#     person_all = json.load(f)
#
# for person in person_all:
#     content = person_all[person]
#     content2 = content.replace('minds', '')
#     current_data.append(content2)
#
# with open(delivery_file_path, 'w') as f:
#     json.dump(current_data, f, ensure_ascii=False, indent=4)

#### TILDES

tildes_path = BASE_PATH/'tildes_final.json'
with open(tildes_path, 'r') as f:
    data = json.load(f)

with open(delivery_file_path, 'r') as f:
    outs = json.load(f)

for d in data:
    content = d['content_summary'] + '\n' + d['content']
    content2 = content.replace("tildes", '').replace("Tildes", '')
    outs.append(content2)

random.shuffle(outs)

with open(delivery_file_path, 'w') as f:
    json.dump(outs, f, indent=2, ensure_ascii=False)


