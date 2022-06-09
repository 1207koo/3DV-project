import os
import glob
from PIL import Image
from tqdm import tqdm
for dir_name in tqdm(glob.glob('hpatches-sequences-release/*')):
    for img_path in glob.glob(os.path.join(dir_name, '*.ppm')):
        Image.open(img_path).save(img_path.replace('.ppm', '.png'))