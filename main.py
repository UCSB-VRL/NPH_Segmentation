#
import os
import argparse
import subprocess
from TestFunc import * 

def imageList(dataPath):
    fileName=[]
    fileList=[]
#     if os.path.isfile(dataPath) and '.nii' in dataPath:
#         fileList+=[dataPath]
#         temp=dataPath
#         if '/' in temp: temp=temp.split('/')[-1]
#         fileName+=[temp.split('.nii')[0]]
#         print(fileName)

    if os.path.isdir(dataPath):
        
        fileList+=[d for d in os.listdir(dataPath) if '.nii' in d]
        
        for temp in fileList:
            fileName+=[temp.split('.nii')[0]]
         
    else:
        raise ValueError('Invalid data path input')
            
    return fileList, fileName

#skull strip
def skull_strip(inName, outName):
    subprocess.call(['bash', 'skull_strip.sh', inName, outName])
    
    print('done')
    


#run test 



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', default='model_backup/epoch49_ResNet2D3Class_2Layer2x2_mixed2_300.pt')
    parser.add_argument('--outputPath', default='reconstructed')
    parser.add_argument('--dataPath', default='data-split/Scans')
    parser.add_argument('--betPath', default='data-split/skull-strip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=200)

    args = parser.parse_args()
    
    
    dataPath=args.dataPath
    modelPath=args.modelPath
    outputPath=args.outputPath
    betPath=args.betPath
    device=args.device
    BS=args.batch_size
    
    fileList, fileName=imageList(dataPath)
    
    for i in range(1):
        
#         skull_strip(os.path.join(dataPath, fileList[i]), os.path.join(betPath, fileName[i]))
        runTest(fileName[i], modelPath,outputPath, dataPath, betPath, device, BS)
    
#             print(fileName)
        
    
    
    
    
    
    
    
    