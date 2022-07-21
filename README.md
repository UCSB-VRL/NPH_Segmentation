# NPH Segmentation
## 😃 Hazel's branch 😃
### from Fei's model
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
### IV. Example output
```
Start Running: Norm_old_003_96yo
Norm_old_003_96yo Elapsed time: 201.69443917274475
------------ Norm_old_003_96yo -------------
middle of 7 slices : 21
max ventricle pos : 21
Correct point: 921933/1172347, Accuracy : 78.63994192845634
    Dice score for class1: 0.7630208055907701
    Dice score for class2: 0.8678160082429361
    Dice score for class3: 0.7963684631485799
    Dice score for class4: 0.5364105560424579

```

### V. Progress

```
1. Use BET result instead of ground truth (manually segmented file) -> done
2. Caculating Dice score with result image -> done
3. Figure out how to do 4 classes segmentation for whole slices, not only 7 slices.
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
### III. Example Result 
What I got from the last epoch :
```
Train Epoch: 99 [200/2834 (7%)]	Train Loss: 0.957378 Current accuracy: 78.623% 
Train Epoch: 99 [400/2834 (14%)]	Train Loss: 0.958383 Current accuracy: 78.532% 
Train Epoch: 99 [600/2834 (21%)]	Train Loss: 0.957505 Current accuracy: 78.605% 
Train Epoch: 99 [800/2834 (28%)]	Train Loss: 0.957999 Current accuracy: 78.528% 
Train Epoch: 99 [1000/2834 (35%)]	Train Loss: 0.958385 Current accuracy: 78.494% 
Train Epoch: 99 [1200/2834 (42%)]	Train Loss: 0.958743 Current accuracy: 78.457% 
Train Epoch: 99 [1400/2834 (49%)]	Train Loss: 0.958633 Current accuracy: 78.474% 
Train Epoch: 99 [1600/2834 (56%)]	Train Loss: 0.958680 Current accuracy: 78.474% 
Train Epoch: 99 [1800/2834 (63%)]	Train Loss: 0.958592 Current accuracy: 78.485% 
Train Epoch: 99 [2000/2834 (71%)]	Train Loss: 0.958938 Current accuracy: 78.447% 
Train Epoch: 99 [2200/2834 (78%)]	Train Loss: 0.959078 Current accuracy: 78.433% 
Train Epoch: 99 [2400/2834 (85%)]	Train Loss: 0.959196 Current accuracy: 78.419% 
Train Epoch: 99 [2600/2834 (92%)]	Train Loss: 0.959362 Current accuracy: 78.399% 
Train Epoch: 99 [2800/2834 (99%)]	Train Loss: 0.959363 Current accuracy: 78.398% 
Train Epoch: 99, Correct point: 1776650/2266428
    Dice score for class1: 0.8066549026865143
    Dice score for class2: 0.7743982847491562
    Dice score for class3: 0.7974527698900845
Val Epoch: 99 [100/642 (15%)]	Test Loss: 0.985823 Current accuracy: 75.306%
Val Epoch: 99 [200/642 (31%)]	Test Loss: 0.986183 Current accuracy: 75.265%
Val Epoch: 99 [300/642 (47%)]	Test Loss: 0.986567 Current accuracy: 75.246%
Val Epoch: 99 [400/642 (62%)]	Test Loss: 0.987213 Current accuracy: 75.205%
Val Epoch: 99 [500/642 (78%)]	Test Loss: 0.987315 Current accuracy: 75.203%
Val Epoch: 99 [600/642 (93%)]	Test Loss: 0.987413 Current accuracy: 75.200%
Val Epoch 99: Correct point: 386033/513344, 75.19967117566388
    Dice score for class1: 0.7695363967847039
    Dice score for class2: 0.7557002972738611
    Dice score for class3: 0.7529083631665691
Test Epoch 99: Correct point: 14869/16992, 87.50588512241055
    Dice score for class1: 0.8970836033700583
    Dice score for class2: 0.8668961560527826
    Dice score for class3: 0.8260303687635575
```
