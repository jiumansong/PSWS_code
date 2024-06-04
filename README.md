# PSWS_code
# A Position-Aware Sets based Weakly Supervised Framework for Whole-Slide Multi-Classification

## Installation
Download the repository and create a project for it.

Then create a conda env and install these packages：
```
conda create -n psws python=3.9 -y
conda activate psws
python setup.py build install
```
| package | version | package | version |
|:------------:|:------------:|:------------:|:------------:|
| openslide-python |4.7.0 | numpy | 1.26.4 |
| setuptools | 65.5.0 | torch | 2.0.1 |
| matplotlib | 3.8.3 | opencv-python | 4.7.0 |
| timm | 0.9.7 | pandas | 2.0.1 | 
| scikit-image | 0.22.0 | umap-learn | 0.5.5 |

## Dataset 
The dataset of the TCGA-SARC project can be download at <https://portal.gdc.cancer.gov/>.
Internal datasets involve patient privacy and are not considered for public release for the time being.

## Data preprocessing
The first step is to process the WSI to generate the position-aware sets, determine the downsampling rate:
```
cd preprocess
python tile.py
```
If staining normalization is required, please execute:
```
cd preprocess
python macenko_picture.py
```

## Feature extraction
```
cd encode
python patch_To_feature.py --device DEVICE --batch_size 1 --class_num NUM
```

## Training and WSI prediction
```
python train.py --EPOCH EPOCH --epoch_step [50] --lr LR --class_num NUM --num_samples BATCH_SIZE --device DEVICE 
```
```
python test.py --EPOCH 1 --class_num NUM --device DEVICE 
```





