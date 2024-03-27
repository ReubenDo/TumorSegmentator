from setuptools import setup, find_packages


setup(name='TumorSegmentator',
        version='1.1.0',
        description='Robust segmentation of tumor in MR images.',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/reubendo/TumorSegmentator',
        author='Reuben Dorent',
        author_email='rdorent@bwh.harvard.edu',
        python_requires='>=3.9',
        license='MIT',
        packages=find_packages(),
        install_requires=[
            'torch>=2.0.0',
            'numpy',
            'psutil',
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'p_tqdm',
            'xvfbwrapper',
            'fury',
            'nnunetv2>=2.2.1',
            # 'requests==2.27.1;python_version<"3.10"',
            # 'requests;python_version>="3.10"',
            'requests',
            'rt_utils',
            'dicom2nifti',
            'pyarrow'
        ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        entry_points={
            'console_scripts': [
                'TumorSegmentator=tumorsegmentator.bin.TumorSegmentator:main',
            ],
        },
    )
