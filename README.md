# NPH Segmentation
Contributors: Fei Xu

This code repository contains Fei's work on the NPH project. 

## Running the Code

This looks like how you can run inference using a trained model
```python
python3 -W ignore main.py --dataPath '/home/fei/documents/GitHub/NPH_new/data-split/Scans' --betPath '/home/fei/documents/GitHub/NPH_new/data-split/Segmentation' --modelPath 'model_backup/epoch35_2Dresnet3Class_wd6_lr2_2Layer2x2_300.pt' --outputPath 'reconstructed2'

```

To train, this looks maybe how to train

```python
python3 -W ignore main.py --dataPath '/home/fei/documents/GitHub/NPH_new/data-split/Scans' --betPath '/home/fei/documents/GitHub/NPH_new/data-split/Segmentation'
```
