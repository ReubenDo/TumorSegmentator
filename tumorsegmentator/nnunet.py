# ADAPTED FROM TotalSegmentator: https://github.com/wasserth/TotalSegmentator/tree/master/totalsegmentator

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from os.path import join
from typing import Union, List
from functools import partial
import tempfile
import inspect
import warnings

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from p_tqdm import p_map
import torch

from tumorsegmentator.libs import nostdout

# nnUNet 2.1
# with nostdout():
#     from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
# nnUNet 2.2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from nnunetv2.utilities.file_path_utilities import get_output_folder

from tumorsegmentator.map_to_binary import class_map
from tumorsegmentator.alignment import as_closest_canonical, undo_canonical
from tumorsegmentator.resampling import change_spacing
from tumorsegmentator.libs import check_if_shape_and_affine_identical
from tumorsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from tumorsegmentator.nifti_ext_header import add_label_map_to_nifti
from tumorsegmentator.postprocessing import keep_largest_blob

# Hide nnunetv2 warning: Detected old nnU-Net plans format. Attempting to reconstruct network architecture...
warnings.filterwarnings("ignore", category=UserWarning, module="nnunetv2")


def _get_full_task_name(task_id: int, src: str="raw"):
    if src == "raw":
        base = Path(os.environ['nnUNet_raw_data_base']) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ['nnUNet_preprocessed'])
    elif src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_fullres"
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir

    # If not found in 3d_fullres, search in 3d_lowres
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_lowres"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    # If not found in 3d_lowres, search in 2d
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "2d"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    raise ValueError(f"task_id {task_id} not found")


def contains_empty_img(imgs):
    """
    imgs: List of image paths
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def supports_keyword_argument(func, keyword: str):
    """
    Check if a function supports a specific keyword argument.

    Returns:
    - True if the function supports the specified keyword argument.
    - False otherwise.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    return keyword in parameters



def nnUNet_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                   trainer="nnUNetTrainerV2", tta=False,
                   num_threads_preprocessing=6, num_threads_nifti_save=2):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    with nostdout():
        from nnunet.inference.predict import predict_from_folder
        from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer

    save_npz = False
    # num_threads_preprocessing = 6
    # num_threads_nifti_save = 2
    # num_threads_preprocessing = 1
    # num_threads_nifti_save = 1
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    disable_tta = not tta
    overwrite_existing = False
    mode = "normal" if model == "2d" else "fastest"
    all_in_gpu = None
    step_size = 0.5
    chk = "model_final_checkpoint"
    disable_mixed_precision = False

    task_id = int(task_id)
    task_name = _get_full_task_name(task_id, src="results")

    # trainer_class_name = default_trainer
    # trainer = trainer_class_name
    plans_identifier = default_plans_identifier

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print("using model stored in ", model_folder_name)

    predict_from_folder(model_folder_name, dir_in, dir_out, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)


def nnUNetv2_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                     trainer="nnUNetTrainer", tta=False,
                     num_threads_preprocessing=3, num_threads_nifti_save=2,
                     plans="nnUNetPlans", device="cuda", quiet=False, step_size=0.5):
    """
    Identical to bash function nnUNetv2_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    dir_in = str(dir_in)
    dir_out = str(dir_out)

    model_folder = get_output_folder(task_id, trainer, plans, model)

    assert device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)  # throws error if setting the second time
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    disable_tta = not tta
    verbose = False
    save_probabilities = False
    continue_prediction = False
    chk = "checkpoint_final.pth"
    npp = num_threads_preprocessing
    nps = num_threads_nifti_save
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0
    allow_tqdm = not quiet


    # nnUNet 2.2.1
    if supports_keyword_argument(nnUNetPredictor, "perform_everything_on_gpu"):
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_gpu=True,  # for nnunetv2<=2.2.1
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
    # nnUNet >= 2.2.2
    else:
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=True,  # for nnunetv2>=2.2.2
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=chk,
    )
    predictor.predict_from_files(dir_in, dir_out,
                                 save_probabilities=save_probabilities, overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp, num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=num_parts, part_id=part_id)



def save_segmentation_nifti(class_map_item, tmp_dir=None, output_path=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if not task_name.startswith("tumor") and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data >= k
    # output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path.format(v))


def nnUNet_predict_image(inputs: List[Union[str, Path, Nifti1Image, None]], 
                         file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False,
                         resample=None, task_name="tumor",
                         save_binary=False, nr_threads_resampling=0, nr_threads_saving=0,
                         output_type="nifti",
                         quiet=False, verbose=True, skip_saving=False,
                         device="cuda"):
    """
    crop: string or a nibabel image
    resample: None or float  (target spacing for all dimensions)
    """
    img_types = []
    for ind, input_ind in enumerate(inputs):
        mod = input_ind[0]
        file_in = input_ind[1]
        if not isinstance(file_in, Nifti1Image) and not file_in is None:
            inputs[ind][1] = Path(file_in)
            img_type = "nifti" if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz") else "dicom"
            if not file_in.exists():
                sys.exit(f"ERROR: The input {mod} file or directory does not exist.")
        else:
            img_type = "nifti"
        img_types.append(img_type)
    
    assert np.all([k=='nifti' for k in img_types]), "Currently only NIFTI are supported"
            
    if file_out is not None:
        file_out = Path(file_out)
    if img_type == "nifti" and output_type == "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose: print(f"tmp_dir: {tmp_dir}")

        for ind, input_ind in enumerate(inputs):
            mod = input_ind[0]
            file_in = input_ind[1]
            img_type = img_types[ind]
            
            if img_type == "dicom":
                if not quiet: print("Converting dicom to nifti...")
                (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
                dcm_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz", verbose=verbose)
                file_in_dcm = file_in
                file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"

                if not quiet: print(f"  found image with shape {nib.load(file_in).shape}")
        
            if isinstance(file_in, Nifti1Image):
                img_in_orig = file_in
            else:
                img_in_orig = nib.load(file_in)
            if len(img_in_orig.shape) == 2:
                raise ValueError("tumorsegmentator does not work for 2D images. Use a 3D image.")
            if len(img_in_orig.shape) > 3:
                print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
                img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:,:,:,0], img_in_orig.affine)

            # takes ~0.9s for medium image
            img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

            img_in = as_closest_canonical(img_in)

            if resample is not None:
                if not quiet: print("Resampling...")
                st = time.time()
                img_in_shape = img_in.shape
                img_in_rsp = change_spacing(img_in, [resample, resample, resample],
                                            order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
                if verbose:
                    print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
                if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
            else:
                img_in_rsp = img_in
                
            nib.save(img_in_rsp, tmp_dir / f"s01_000{ind}.nii.gz")


        step_size = 0.5

        st = time.time()
        if not quiet: print("Predicting...")
        with nostdout(verbose):
            # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
            #                nr_threads_resampling, nr_threads_saving)
            nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                nr_threads_resampling, nr_threads_saving,
                                device=device, quiet=quiet, step_size=step_size)

        if not quiet: print(f"  Predicted in {time.time() - st:.2f}s")

        # Combine image subparts back to one image
        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        

        if resample is not None:
            if not quiet: print("Resampling...")
            if verbose: print(f"  back to original shape: {img_in_shape}")
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                      order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                      force_affine=img_in.affine)

        if verbose: print("Undoing canonical...")
        img_pred = undo_canonical(img_pred, img_in_orig)


        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)
        if save_binary:
            img_data = (img_data > 0).astype(np.uint8)


        # Prepare output nifti
        # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
        # float or int and therefore the masks will need a lot more space.
        # (infos on header: https://nipy.org/nibabel/nifti_images.html)
        new_header = img_in_orig.header.copy()
        new_header.set_data_dtype(np.uint8)
        img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
        img_out = add_label_map_to_nifti(img_out, class_map[task_name])

        if file_out is not None and skip_saving is False:
            if not quiet: print("Saving segmentations...")

            # Select subset of classes if required
            selected_classes = class_map[task_name]
            if output_type == "dicom":
                file_out.mkdir(exist_ok=True, parents=True)
                save_mask_as_rtstruct(img_data, selected_classes, file_in_dcm, file_out / "segmentations.dcm")
            else:
                st = time.time()
                if img_type == "nifti":
                    if '.nii' in file_out.name: # output is a filename
                        dir_out = file_out.parent
                        base_output_flnm = file_out.name
                    else: # output is a folder
                        dir_out = file_out
                        if not isinstance(file_in, Nifti1Image):
                            base_output_flnm = str(file_in.stem).replace('.nii','')
                            base_output_flnm += "_{}.nii.gz"
                        else:
                            base_output_flnm = "output_{}.nii.gz"

                            
                    dir_out.mkdir(exist_ok=True, parents=True)
                        
                if np.prod(img_data.shape) > 512*512*1000:
                    print("Shape of output image is very big. Setting nr_threads_saving=1 to save memory.")
                    nr_threads_saving = 1
                
                # Code for single threaded execution  (runtime:24s)
                if nr_threads_saving == 1:
                    for k, v in selected_classes.items():
                        binary_img = img_data == k
                        output_path = str(dir_out / base_output_flnm.format(v))
                        nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, new_header), output_path)
                else:
                    # Code for multithreaded execution
                    #   Speed with different number of threads:
                    #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                    nib.save(img_pred, tmp_dir / "s01.nii.gz")
                    output_path = str(dir_out / base_output_flnm)
                    _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, output_path=output_path, nora_tag="None", header=new_header, task_name=task_name, quiet=quiet),
                            selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

            if not quiet: print(f"  Saved in {time.time() - st:.2f}s")


    return img_out

