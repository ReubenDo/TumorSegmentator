#!/usr/bin/env python
import sys
import os
import argparse
from pkg_resources import require
from pathlib import Path

from tumorsegmentator.python_api import tumorsegmentator


def main():
    parser = argparse.ArgumentParser(description="Segment skin surface in MR images.",
                                     epilog="Written by Reuben Dorent")

    parser.add_argument("-t1", metavar="filepath", dest="t1",
                        help="T1 nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=False, 
                        default=None)

    parser.add_argument("-cet1", metavar="filepath", dest="cet1",
                        help="Contrast-enhanced T1 nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=False,
                        default=None)

    parser.add_argument("-t2", metavar="filepath", dest="t2",
                        help="T2 nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=False,
                        default=None)

    parser.add_argument("-flair", metavar="filepath", dest="flair",
                        help="FLAIR nifti image or folder of dicom slices",
                        type=lambda p: Path(p).absolute(), required=False,
                        default=None)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-ot", "--output_type", choices=["nifti", "dicom"],
                    help="Select if segmentations shall be saved as Nifti or as Dicom RT Struct image.",
                    default="nifti")


    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations",
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model (3mm)",
                        default=False)
    

    # cerebral_bleed: Intracerebral hemorrhage
    # liver_vessels: hepatic vessels
    parser.add_argument("-ta", "--task", choices=["skin"],
                        # future: liver_vessels, head,
                        help="Select which model to use. This determines what is predicted.",
                        default="skin")


    # "mps" is for apple silicon; the latest pytorch nightly version supports 3D Conv but not ConvTranspose3D which is
    # also needed by nnU-Net. So "mps" not working for now.
    # https://github.com/pytorch/pytorch/issues/77818
    parser.add_argument("-d", "--device", choices=["gpu", "cpu", "mps"],
                        help="Device to run on (default: gpu).",
                        default="gpu")

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    # Tests:

    parser.add_argument('--version', action='version', version=require("TumorSegmentator")[0].version)

    args = parser.parse_args()

    tumorsegmentator(t1=args.t1, 
                    cet1=args.cet1, 
                    t2=args.t2, 
                    flair=args.flair, 
                    output=args.output, 
                    nr_thr_resamp=args.nr_thr_resamp, 
                    nr_thr_saving=args.nr_thr_saving,
                    task=args.task,
                    output_type=args.output_type, 
                    quiet=args.quiet, 
                    verbose=args.verbose, 
                    skip_saving=False,
                    device=args.device)
    
if __name__ == "__main__":
    main()
