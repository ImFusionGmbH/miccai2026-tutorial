# ImFusion Python SDK tutorial, MICCAI 2026

Two talks, given as RISE slideshows from Jupyter notebooks: *Rapid prototyping with
the ImFusion Python SDK* and *From Data to Models*.

```
rapid_prototyping.ipynb        the first talk
from_data_to_models.ipynb      the second talk, on the ML lifecycle
brain_extraction_algorithm.py  the data component and algorithm of the first talk, for the ImFusion Suite
```

## Setup

```bash
cd miccai2026-tutorial/rapid_prototyping_and_from_data_to_models
uv sync
```
You can also use `uv sync --group specialty` to add the modality-specific modules that the last section of the first
talk uses.


## Rapid Prototyping

`rapid_prototyping.ipynb` follows one neurosurgical case, a preoperative MR and an
intraoperative 3D ultrasound, from files on disk to a single ImFusion file.

- **00 setup**: installing the SDK and activating the license.
- **01 load**: a DICOM series, a NIfTI volume, meshes and point clouds through one function,
  with their metadata and coordinates converted to LPS.
- **02 see**: `imfusion.show` on volumes, frames, meshes and point clouds.
- **03 process**: MR-to-US registration with LC2, calling any SDK algorithm by name, bias
  field correction.
- **04 extend**: compatibility with existing packages and models through numpy and torch conversions.
- **05 store**: a custom data component for the QC result, and the whole case in one `.imf`
  file.
- **06 see (even more)**: the brain extraction registered as an ImFusion algorithm and run in
  the ImFusion Suite.
- **07 process (even more)**: an overview of the specialty modules, with optional demos of full
  spine segmentation and CT-to-X-ray 2D/3D registration.

The case is in `data/`: the MR as a DICOM series, the ultrasound as NIfTI, a ventricle
segmentation, and a brain mesh and point cloud. `data/spine_ct.imf` is the CT for section 07.


## From Data to Models

`from_data_to_models.ipynb` is the second talk. Its four
sections cover the BraTS-Africa project in the Labels Database view, a
`machinelearning.Dataset` backed by that project, reviewing saved model predictions and their
Dice scores in Labels, and deploying a model with `machinelearning.MachineLearningModel`.

The data lives in `data/brats-africa`, see its `README.md` for the license and citation. The
setup cell downloads the images from TCIA on the first run. Section 03 writes the saved
predictions and Dice tags into the Labels project there, so git shows the project as changed
afterwards.

Section 04 loads a model from `libimfusion-cranial`, part of the `specialty` group; the license
must include the `Cranial` module.
