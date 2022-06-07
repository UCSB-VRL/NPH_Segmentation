
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
import nibabel as nib
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt




device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
# device='cpu'
BS=100
def getCenter(image, segmentation, i, j, k):

    sample=image[i-16:i+16+1,j-16:j+16+1,k-1:k+1+1]
    center=segmentation[i:i+1+1,j:j+1+1,k]
    
    return sample, center

def readAll(imgPath, maskPath):
    
    positions=[]
    
    image = nib.load(imgPath).get_fdata()

    annotation = nib.load(maskPath).get_fdata()
    
    x,y,z=image.shape
    
    for z in range(image.shape[2]):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):

                if annotation[x,y,z]==3 or  annotation[x,y,z]==6:
                    annotation[x,y,z]=1

                if annotation[x,y,z]==5:
                    annotation[x,y,z]=2

                if annotation[x,y,z]==4:
                    annotation[x,y,z]=3


                if image[x,y,z]>200: image[x,y,z]=200
                if image[x,y,z]<-100: image[x,y,z]=-100
    
    image+=100
    image=image/300
    
    for k in range(1, z-1, 1):
        for i in range(17, x-17, 2):
            for j in range(17, y-17, 2):
                
                
                sample, center =getCenter(image, annotation, i, j, k)
                if center.any():
                    positions.append((i,j,k))
#     return image, annotation
    return image, annotation, positions, image.shape
    

def getPatch(image_full, seg_full, i, j, k):
    
    image, segmentation =getCenter(image_full, seg_full, i, j, k)    
        
    
    return image, segmentation, torch.tensor([i,j,k])


class NPHDataset(Dataset):
    def __init__(self, dataPath, segPath, name, Train=False):
        
        self.name=name
        self.image, self.annotation, self.imgList, self.imageShape=readAll(dataPath, segPath)
        self.transform=transforms.ToTensor()

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        
#         return 0
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i,j,k=self.imgList[idx]
        data, annotation, pos=getPatch(self.image, self.annotation, i, j, k)
    
        image = self.transform(data)
        sample = {'img': image,
                  'label': annotation,
                  'pos': pos
                 }
        return sample


# In[6]:

print('----Load model----')
ResNet=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
class MyModel(nn.Module):
    def __init__(self,ResNet, num_classes=4, num_outputs=9):
        super(MyModel, self).__init__()
 
        self.layer0=nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
            
        )
        
        self.layer1=ResNet.layer1
        self.layer2=ResNet.layer2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc=nn.Linear(in_features=128, out_features=num_classes*num_outputs, bias=True)
        
    def forward(self, x):

        x=self.layer0(x)
        x=self.layer1(x)        
        x=self.layer2(x) 
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)              
        return x
model = MyModel(ResNet, num_classes=4, num_outputs=4).to(device)
model.load_state_dict(torch.load('model_backup/epoch99_ResNet2D3Class_2Layer2x2_mixed2_300.pt',map_location=device))
modelname='2Layer2x2_mixed2_300_epoch99'



# def dice(TP, FN, FP, pred, target):
#     correct=0
#     for i in range(target.shape[0]):
#         for j in range(target.shape[1]):
#             if pred[i,j].view_as(target[i,j])==target[i,j]:
#                 TP[int(target[i,j])]+=1

#                 correct+=1

#             else:
#                 FN[int(target[i,j])]+=1
#                 FP[int(pred[i,j])]+=1
                
#     return correct


# In[9]:


# def evaluation(output, target, TP, FP, FN):
#     total=0
#     correct=0
    
#     criteria = nn.CrossEntropyLoss()
#     loss=criteria(output, target.long())        
#     pred=output.argmax(dim=1, keepdim=True)    
#     N=output.shape[0]
#     for k in range(N):

#         correct+=dice(TP, FN, FP, pred[k,0, :,:], target[k,:,:])
    
#         total+=target.shape[1]*target.shape[2]

#     return pred, loss, correct, total


# In[14]:


def test(test_loader, shape):

    model.eval()
#     testLoss = 0
#     testCorrect = 0
#     testTotal=0

#     result=[]
#     TP=[0]*7
#     FP=[0]*7
#     FN=[0]*7

    # Don't update model
    print(len(test_loader))
    with torch.no_grad():
        predUnique={}
        targetUnique={}
        # Predict
        
        reconstructed=np.zeros(shape)
        probScore=np.zeros((4, shape[0], shape[1],shape[2]))
        for batch_index, batch_samples in enumerate(test_loader):
            data, target, pos = batch_samples['img'].to(device, dtype=torch.float), batch_samples['label'].to(device), batch_samples['pos']
            output = model(data)
            softmax=nn.Softmax(dim=1)
            output=torch.reshape(output,(output.shape[0], 4, 2, 2))
            
            output=softmax(output)
            pred=output.argmax(dim=1, keepdim=True)

#             pred, loss, correct, total=evaluation(output, target, TP, FP, FN)
            N=output.shape[0]

            for k in range(N):

                x, y, z=pos[k][0].item(), pos[k][1].item(), pos[k][2].item()

                reconstructed[x:x+1+1,y:y+1+1,z]=pred[k,0,:,:].cpu()
                
#                 probScore[:, x-1:x+1,y-1:y+1,z]=output[k,:, :,:].cpu()
                
#             testCorrect+=correct
#             testTotal+=total
            
#             if (batch_index+1) % (50) == 0:
#                 print('[{}({:.0f}%)]'.format(batch_index+1,(batch_index+1)/len(test_loader)*100), end=' ')
#     return testCorrect, testTotal, TP, FP, FN, reconstructed
    return reconstructed

def runTest(imgName):

    dataPath=os.path.join('data-split/Scans','{}.nii.gz'.format(imgName))
    segPath=os.path.join('data-split/Segmentation','Final_{}.nii.gz'.format(imgName))

    testDataset=NPHDataset(dataPath,segPath,imgName,Train=False)
    test_loader = DataLoader(testDataset, batch_size=BS, num_workers=8, drop_last=False, shuffle=False)
    shape=testDataset.imageShape




    # In[15]:
    print('----Start Running----')
    import time

    start = time.time()

#     testCorrect, testTotal, TP, FP, FN, reconstructed=test(test_loader, shape)
    reconstructed=test(test_loader, shape)
    
    correct, total, TP, FP, FN=diceScore(reconstructed, testDataset.annotation)
    
    print(modelname, 'on', imgName)
    print('Correct point: {}/{}, {}'.format(correct, total, correct/total*100))   
    for i in range(1,4):
        print('    Dice score for class{}: {}'.format(i, 2*TP[i]/(2*TP[i]+FP[i]+FN[i])))    
        
    img = nib.Nifti1Image(reconstructed, np.eye(4))
    nib.save(img, 'reconstructed/reconstructed_{}_{}.nii.gz'.format(modelname, imgName))  
    print('Save to: reconstructed_{}_{}.nii.gz'.format(modelname, imgName))

    result_noNoise=eliminateNoise(reconstructed)                
    correct, total, TP, FP, FN=diceScore(result_noNoise, testDataset.annotation)
        
    # In[16]:
    img = nib.Nifti1Image(reconstructed, np.eye(4))
    nib.save(img, 'reconstructed/elimNoise_reconstructed_{}_{}.nii.gz'.format(modelname, imgName))  
    print('Save to: elimNoise_reconstructed_{}_{}.nii.gz'.format(modelname, imgName))

    
    print('{} on {} after noise cancellation'.format(modelname,imgName))
    print('Correct point: {}/{}, {}'.format(correct, total, correct/total*100))   
    for i in range(1,4):
        print('    Dice score for class{}: {}'.format(i, 2*TP[i]/(2*TP[i]+FP[i]+FN[i])))    

    end = time.time()
    print('Elapsed time:', end - start)


# In[ ]:

def eliminateNoise(label):
    neighbors=[(-1,0),(1,0),(0,-1),(0,1)]
                
    seen=set()
    import heapq
    position=[]
    heapq.heapify(position)

    island=0
    newLabel=np.zeros(label.shape)
    i, j, k=label.shape
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]!=0) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))


                    while position:
                        cur=heapq.heappop(position)
                        area+=1


                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
#                             if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]]==label[x,y,z] and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]))



                    for (posX, posY, posZ) in curIsland: 
                        if area<16:
                            newLabel[posX, posY, posZ]=2
                        else:
                            newLabel[posX, posY, posZ]=label[x,y,z]


    return newLabel

def diceScore(initial, final):
    correct=0
    total=0
    TP=[0]*4
    FP=[0]*4
    FN=[0]*4
    
    for i in range(initial.shape[0]):
        for j in range(initial.shape[1]):
            for k in range(initial.shape[2]):
                if final[i,j,k]==0 and initial[i,j,k]==0: continue
                total+=1
                if initial[i,j,k]==final[i,j,k]:
                    TP[int(final[i,j,k])]+=1

                    correct+=1

                else:
                    FN[int(final[i,j,k])]+=1
                    FP[int(initial[i,j,k])]+=1

    return correct, total, TP, FP, FN

  

imgName='NPH_shunt_002_70yo'

runTest(imgName)
