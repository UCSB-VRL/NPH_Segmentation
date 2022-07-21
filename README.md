# NPH Segmentation
## 😃 Hazel's branch 😃
### originally from Fei's module
---

# Test code
### I. How to run code

```python
python3 main.py
```

### II. Basic Description 

**What we need** : data-split/Scans/{filename}.nii.gz

**Skull stripping result will be stored in** : data-split/skull-strip/{filename}_Mask.nii.gz

**Final segment result will be stored in** : reconstructed/ventreconstructed_{filename}.nii.gz


### III. Brief Step by Step Description

```
1. Do skull stripping 
2. Get reconstructed_{filename}.nii.gz which only can distinguish CSF from ventricle and subarachnoid
3. Find a slice number which has a maximum area of [ventricle + subarachnoid]
4. For 7 slices based on max-area slide, distinguish subarachnoid from ventricle by connecting subarachnoid to skull (in 3D)
```

### IV. Progress

```
1. Use BET result instead of ground truth (manually segmented file) -> done
2. Caculating Dice score with result image -> on progress
3. Future work will be focused on figuring out how to do 4 classes segmentation for whole slices, not only 7 slices.
```

# Training Code
### I. How to run the code
```python
python3 ResNet2Layer2x2_norm_blurnoise_newdata.py
```

### II. What you need
```
Segmentation_patch_test
Segmentation_patch_mixed_6
Scans_patch_mixed_6
Scan_patch_test

val_positions_mixed_6.txt
train_positions_mixed_6.txt
test_positions_rand.txt
image_shape.txt

Scan_patch folders have imageNorm_subjectInfo_slideNum.npy
Segmentation patch folders have labelNorm_subjectInfo_slideNum.npy

(You can find example files in trainingData_examples folder)
```
