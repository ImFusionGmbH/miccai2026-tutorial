# BraTS-Africa 2024

ImFusion Labels project of the BraTS-Africa collection: pre-operative multi-parametric brain MRI of
146 patients from Nigeria (95 glioma, 51 other neoplasms), with expert tumor annotations.

The repository holds the project without its images: `project/project.xml`, the ground truth label
maps in `project/labels` and the thumbnails. The images are fetched from TCIA and placed next to the
project by

```bash
python data/brats-africa/fetch.py
```

`SHA256SUMS` lists the checksums of the 584 images the project references.

## Predictions

`predictions` holds one label map per validation case, as a model would output them: a gzipped
`torch.save` of a `uint8` tensor of shape [1, 1, D, H, W] on the grid of the T2 image, with values 1
(necrotic core), 2 (peritumoral edema) and 3 (enhancing tumor). The tensors carry no spacing or
orientation; section 03 of `from_data_to_models.ipynb` restores them from the GT with
`SharedImageSet.from_torch(..., get_metadata_from=...)`. They were produced by the tumor
segmentation model of the ImFusion Cranial plugin.

Section 03 of `from_data_to_models.ipynb` adds them to the project as a `Prediction` layer, together
with the Float tags `Dice_NCR`, `Dice_ED` and `Dice_ET`. To restore the committed project:

```bash
git checkout data/brats-africa/project && git clean -fd data/brats-africa/project
```

## License and citation

The data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The label maps
and thumbnails in `project` are derived from it. Use of the data must follow the
[TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)
and cite:

Adewole, M., Rudie, J.D., Gbadamosi, A., Zhang, D., Raymond, C., Ajigbotoshso, J., Toyobo, O.,
Aguh, K., Omidiji, O., Akinola R., Suwaid, M.A., Emegoakor, A., Ojo, N., Kalaiwo, C., Babatunde,
G., Ogunleye, A., Gbadamosi, Y., Iorpagher, K., Onuwaje M., Betiku B., Saluja, R., Menze, B., Baid,
U., Bakas, S., Dako, F., Fatade A., Anazodo, U.C. (2024) Expanding the Brain Tumor Segmentation
(BraTS) data to include African Populations (BraTS-Africa) (version 1) [Dataset]. The Cancer
Imaging Archive. https://doi.org/10.7937/v8h6-8x67

Collection page: https://www.cancerimagingarchive.net/collection/brats-africa/
