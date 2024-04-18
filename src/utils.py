import os
from rembg import remove

def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if not os.path.isdir(path):
            raise

def rm_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)