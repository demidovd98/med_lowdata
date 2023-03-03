# Datasets

Please download the datasets from the official corresponding websites and make sure they contain the below-listed files and are organised in the below-mentioned corresponding ways.


<hr />


## CRC:

[Download page](https://zenodo.org/record/53169#.ZAE0WXbP1Eb).

Directory tree:
```bash
CRC_colorectal_cancer_histology/
|–– Kather_texture_2016_image_tiles_5000/ # contains folders with images
|   |–– all/ # all images of all classes in one folder
|   |   |–– 1A1F_CRC-Prim-HE-08_024.tif_Row_1651_Col_151.tif
|   |   |–– 1A3D_CRC-Prim-HE-02_copy.tif_Row_151_Col_301.tif
|   |   |–– ...
|–– low_data/ # contains files with split information
|   |–– 50_50/ # 50/50 train/test split
|   |   |–– 3way_splits/ # 3 randomly drawn data splits
|   |   |   |–– 1
|   |   |   |   |–– train_1_sp1.txt
|   |   |   |   |–– train_3_sp1.txt
|   |   |   |   |–– ...
|   |   |   |–– 2
|   |   |   |   |–– train_1_sp2.txt
|   |   |   |   |–– train_3_sp2.txt
|   |   |   |   |–– ...
|   |   |   |–– 3
|   |   |   |   |–– train_1_sp3.txt
|   |   |   |   |–– train_3_sp3.txt
|   |   |   |   |–– ...
```


<hr />


