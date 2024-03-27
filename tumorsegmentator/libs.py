# ADAPTED FROM TotalSegmentator: https://github.com/wasserth/TotalSegmentator/tree/master/totalsegmentator

import io
import os
import contextlib
import sys
import random
import json
import time
import string
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm
import requests
import numpy as np
import nibabel as nib

from tumorsegmentator.map_to_binary import class_map
from tumorsegmentator.config import get_tumorseg_dir, get_weights_dir

"""
Helpers to suppress stdout prints from nnunet
https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
"""
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield


def download_url_and_unpack(url, config_dir):

    tempfile = config_dir / "tmp_download_file.zip"

    try:
        st = time.time()
        with open(tempfile, 'wb') as f:
            # session = requests.Session()  # making it slower

            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                # With progress bar
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()

        print("Download finished. Extracting...")
        # call(['unzip', '-o', '-d', network_training_output_dir, tempfile])
        with zipfile.ZipFile(config_dir / "tmp_download_file.zip", 'r') as zip_f:
            zip_f.extractall(config_dir)
        # print(f"  downloaded in {time.time()-st:.2f}s")
    except Exception as e:
        raise e
    finally:
        if tempfile.exists():
            os.remove(tempfile)


def download_pretrained_weights(task_id):

    config_dir = get_weights_dir()
    config_dir.mkdir(exist_ok=True, parents=True)
    # (config_dir / "3d_fullres").mkdir(exist_ok=True, parents=True)
    # (config_dir / "3d_lowres").mkdir(exist_ok=True, parents=True)
    # (config_dir / "2d").mkdir(exist_ok=True, parents=True)

    # url = "http://backend.totalsegmentator.com"
    url = "https://github.com/reubendo/tumorsegmentator/releases/download"

    if task_id == 501:
        weights_path = config_dir / "Dataset501_BraTST1ceT1"
        # WEIGHTS_URL = "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
        # WEIGHTS_URL = url + "/static/totalseg_v2/Dataset291_TotalSegmentator_part1_organs_1559subj.zip"
        WEIGHTS_URL = url + "/v1.0.0-weights/Dataset501_BraTST1ceT1.zip"
    elif task_id == 502:
        weights_path = config_dir / "Dataset502_BraTST1ceT1T2FLAIR"
        # WEIGHTS_URL = "https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
        # WEIGHTS_URL = url + "/static/totalseg_v2/Dataset291_TotalSegmentator_part1_organs_1559subj.zip"
        WEIGHTS_URL = url + "/v1.0.0-weights/Dataset502_BraTST1ceT1T2FLAIR.zip"

    else:
        raise ValueError(f"For task_id {task_id} no download path was found.")

    if not weights_path.exists():

        download_url_and_unpack(WEIGHTS_URL, config_dir)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    nib.save(nib.Nifti1Image(img_out, ref_img.affine), multilabel_file)



def compress_nifti(file_in, file_out, dtype=np.int32, force_3d=True):
    img = nib.load(file_in)
    data = img.get_fdata()
    if force_3d and len(data.shape) > 3:
        print("Info: Input image contains more than 3 dimensions. Only keeping first 3 dimensions.")
        data = data[:,:,:,0]
    new_image = nib.Nifti1Image(data.astype(dtype), img.affine)
    nib.save(new_image, file_out)


def check_if_shape_and_affine_identical(img_1, img_2):

    if not np.array_equal(img_1.affine, img_2.affine):
        print("Affine in:")
        print(img_1.affine)
        print("Affine out:")
        print(img_2.affine)
        print("Diff:")
        print(np.abs(img_1.affine-img_2.affine))
        print("WARNING: Output affine not equal to input affine. This should not happen.")

    if img_1.shape != img_2.shape:
        print("Shape in:")
        print(img_1.shape)
        print("Shape out:")
        print(img_2.shape)
        print("WARNING: Output shape not equal to input shape. This should not happen.")


