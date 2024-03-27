
# ADAPTED FROM TotalSegmentator: https://github.com/wasserth/TotalSegmentator/tree/master/totalsegmentator

from pathlib import Path
from typing import Union
from nibabel.nifti1 import Nifti1Image
import torch
from tumorsegmentator.libs import download_pretrained_weights
from tumorsegmentator.config import setup_nnunet, setup_tumorseg
from tumorsegmentator.map_to_model import model_map

def tumorsegmentator(t1: Union[str, Path, Nifti1Image, None], 
                     cet1: Union[str, Path, Nifti1Image, None], 
                     t2: Union[str, Path, Nifti1Image, None], 
                     flair: Union[str, Path, Nifti1Image, None], 
                     output: Union[str, Path, None]=None, 
                     nr_thr_resamp=1, nr_thr_saving=6,
                     task="tumor",
                     output_type="nifti", quiet=False, verbose=False,
                     skip_saving=False, device="gpu"):
    """
    Run tumorsegmentator from within python.

    For explanation of the arguments see description of command line
    arguments in bin/tumorsegmentator.

    Return: multilabel Nifti1Image
    """
    if not isinstance(t1, Nifti1Image) and not t1 is None:
        t1 = Path(t1)
    if not isinstance(cet1, Nifti1Image) and not cet1 is None:
        cet1 = Path(cet1)
    if not isinstance(t2, Nifti1Image) and not t2 is None:
        t2 = Path(t2)
    if not isinstance(flair, Nifti1Image) and not flair is None:
        flair = Path(flair)

    if output is not None:
        output = Path(output)
    else:
        skip_saving = True
        # raise ValueError("Output path is required.")

    # available devices: gpu | cpu | mps
    if device == "gpu": device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU detected. Running on CPU. This can be very slow. The '--fast' option can help to reduce runtime.")
        device = "cpu"

    setup_nnunet()
    setup_tumorseg()

    from tumorsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value
    
    # Order matters here:
    inputs = []
    if not t1 is None:
        inputs.append(['t1', t1])
    if not cet1 is None:
        inputs.append(['cet1',cet1])
    if not t2 is None:
        inputs.append(['t2', t2])
    if not flair is None:
        inputs.append(['flair', flair])
        
    model_key = '_'.join([k[0] for k in inputs])
    assert model_key in model_map.keys(), f'Incorrect set of images, should have {model_map.keys()}'
    task_id = model_map[model_key]
    resample = None
    trainer = "nnUNetTrainer"
    model = "3d_fullres"
    folds = ['all']     


    # fast statistics are calculated on the downsampled image
    download_pretrained_weights(task_id)

    seg_img = nnUNet_predict_image(inputs, output, task_id, model=model, folds=folds,
                            trainer=trainer, tta=False, resample=resample,
                            task_name="tumor",
                            nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving,
                            output_type=output_type,
                            quiet=quiet, verbose=verbose, skip_saving=skip_saving, device=device)

    return seg_img
