# NPH Segmentation
Contributors: Fei Xu

This code repository contains Fei's work on the NPH project. 

## Running the Code

This looks like how you can run inference using a trained model
```python
python3 -W ignore main.py --dataPath '/home/fei/documents/GitHub/NPH_new/data-split/Scans' --betPath '/home/fei/documents/GitHub/NPH_new/data-split/Segmentation' --modelPath 'model_backup/epoch35_2Dresnet3Class_wd6_lr2_2Layer2x2_300.pt' --outputPath 'reconstructed2'

```

To train the `ResNet2Layer2x2_norm_blurnoise`:

```python
python3 ResNet2Layer2x2_norm_blurnoise_newdata-Copy1.py
```
```
To run this code without any modification, what you need are:

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

**You can find example files in trainingData_examples folder**
```

__Sample Output from Training__
```
Using cache found in /home/fei/.cache/torch/hub/pytorch_vision_v0.10.0
/home/fei/.local/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Train Epoch: 0 [200/11333 (2%)]	Train Loss: 1.300252 Current accuracy: 44.305% 
```

__Skull Strip__
```sh
bash skull_strip.sh data-split/Scans/Norm_old_005_64yo.nii.gz data-split/skull-strip/Norm_old_005_64yo
```

### Notes
Diff of two files
```
vimdiff ResNet2Layer2x2_norm_blurnoise_newdata-Copy1.py ResNet1Layer2x2_norm_blurnoise_newdata.py
```
