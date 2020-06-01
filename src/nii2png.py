#################################
# Importing libraries
#################################
import os # OS informations
import matplotlib.pyplot as plt # Plot images
from tqdm import tqdm # Status bar
import nibabel as nib # Load .nii files
import gc # Garbage collector
import warnings # Supress warnings
import argparse # Script arguments

warnings.simplefilter('ignore')

parse = argparse.ArgumentParser()
parse.add_argument('--disease', '-d', required=True, help='Type of disease to convert .nii files. Must be "AD", "MCI" or "CN"', choices=['AD', 'MCI', 'CN'])
args = parse.parse_args()

SRC_DIR = os.path.join(os.path.abspath('.'), 'src')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
FINAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'dataset', args.disease)

try:
    os.makedirs(FINAL_DATA_DIR)
except:
    print(f'Path <{FINAL_DATA_DIR}> already exists!')

#################################
# Converting .nii to png
#################################
files = []

print('Searching .nii files')

for (root, dirs, file) in os.walk(os.path.join(DATA_DIR, args.disease)):
    if len(file) == 1:
        files.append(os.path.join(root, file[0]))

print('Ok.')
print(f'{len(files)} .nii files found.')

gc.collect()

print('\nLoading slices and converting to .png')
for file in tqdm(files):

    id_session = os.path.dirname(file)
    id_data = os.path.dirname(id_session)
    exam_type = os.path.dirname(id_data)
    id_pacient = os.path.dirname(exam_type)
    
    image = nib.load(file).get_data()

    for slice_ in tqdm(range(image.shape[2])):
        fig, ax = plt.subplots(dpi=300)
        ax.imshow(image[:, :, slice_][:, :, 0], cmap='bone')
        ax.axis('off')
        ax.axis('tight')
        ax.axis('image')
        img_name = ' '.join(
            [os.path.basename(id_pacient), os.path.basename(exam_type), os.path.basename(id_data)[:-2], os.path.basename(id_session)]
        )

        fig.savefig(os.path.join(FINAL_DATA_DIR, img_name + f' SLICE_{slice_ + 1}.png'))
        plt.close(fig)
        gc.collect()
    print('Ok.')
