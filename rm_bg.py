import os
from tqdm import tqdm
from src.utils import rm_background

## os.walk in data directory and find .jpg files
for root, dirs, files in tqdm(os.walk('data')):
    for file in tqdm(files):
        if file.endswith('.jpg'):
            input_path = os.path.join(root, file)
            ## output_path is same as input_path but with background removed named 'no_bg'
            output_path = os.path.join(root, 'no_bg_' + file)
            ## remove background
            rm_background(input_path, output_path)
