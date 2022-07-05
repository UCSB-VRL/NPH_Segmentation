import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import copy
import heapq

def connectToBoundary(label, classIdx, tolerance):
    neighbors=[]
    for i in range(-1, 2):
        for j in range(-1, 2):
            k=0
            neighbors.append((i,j,k))
                
    seen=set()
    
    position=[]
    heapq.heapify(position)

    island=0
    newLabel=np.zeros(label.shape)
    i, j, k=label.shape
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]==classIdx) and (x,y,z) not in seen:
                    island+=1
                    area=0
                    curIsland=set()
                    seen2=set()
                    seen.add((x,y,z))
                    curIsland.add((x,y,z))
                    heapq.heappush(position, (x,y,z))

                    connected=False
                    while position:
                        cur=heapq.heappop(position)
                    
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if (label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==classIdx) and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2], 0))

                    position2=[]
                    heapq.heapify(position2)
                    
                    for cur in curIsland:
                        heapq.heappush(position2,(cur[0],cur[1],cur[2],0))
                        seen2.add(cur)
                    while position2:
                        cur=heapq.heappop(position2)
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    
                            if (label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]!=0) and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen2 and cur[3]<tolerance:
                                seen2.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position2, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2], cur[3]+1))
                            
                            elif label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==0:
                                connected=True
                                break   
                        
                    if connected:

                        for (posX, posY, posZ) in curIsland: 

                            label[posX, posY, posZ]=3
                
def maxArea(label, classIdx, connectivity=8, findMax=True):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                k=0
                neighbors.append((i,j,k))
    elif connectivity==4:
        neighbors=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]
        
    else:
        
        return
                
    seen=set()
    
    position=[]
    heapq.heapify(position)
    islandDict={}
    
    maxArea=0
    maxPos=(0,0,0)
    island=0
    newLabel=copy.deepcopy(label)
    i, j, k=label.shape
    
    for z in range(k):
        for x in range(i):
            for y in range(j):
                
                if (label[x,y,z]==classIdx) and (x,y,z) not in seen:
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
                            if cur[2]-neighbor[2]<0 or cur[2]-neighbor[2]>=k: continue    

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]]==label[x,y,z] and (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                curIsland.add((cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1],cur[2]-neighbor[2]))
                    
                    islandDict[(x,y,z)]=frozenset(curIsland)
#                     print(island, area)
                    
                    if findMax:
                        if area>maxArea: 
                            maxArea=area
                            maxPos=(x,y,z)

    return islandDict[maxPos], maxArea, maxPos

def Connectivity(label, classIdx, targetIdx, refClass=1,connectivity=8):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbors.append((i,j))
    elif connectivity==4:
        neighbors=[(1,0),(-1,0),(0,1),(0,-1)]
        
    else:
        
        return
                
    seen=set()
    
    island=0
    position=[]
    heapq.heapify(position)

    i, j=label.shape
    
    for x in range(i):
        for y in range(j):

            if (label[x,y]==refClass) and (x,y) not in seen:
                island+=1
                seen.add((x,y))
                heapq.heappush(position, (x,y))

                while position:
                    cur=heapq.heappop(position)

                    for neighbor in neighbors:

                        if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                        if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue

                        if label[cur[0]-neighbor[0],cur[1]-neighbor[1]]==classIdx and (cur[0]-neighbor[0],cur[1]-neighbor[1]) not in seen:
                            seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1]))
                            label[cur[0]-neighbor[0],cur[1]-neighbor[1]]=targetIdx
                            heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1]))

                

def numIsland(label,connectivity=8):
    neighbors=[]
    if connectivity==8:
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbors.append((i,j))
    elif connectivity==4:
        neighbors=[(1,0),(-1,0),(0,1),(0,-1)]
        
    else:
        
        return
               
    seen=set()
    
    island=0
    position=[]
    heapq.heapify(position)

    i, j=label.shape
    
    
    for y in range(j):
        for x in range(i-1,-1,-1):

            if (label[x,y]!=0) and (x,y) not in seen:
                
                if island==1:
                    if area>100: 
                        island+=1
                        break
                        
                    else: island=0

                if island==0:
                    island+=1
                    area=0
                    seen.add((x,y))
                    heapq.heappush(position, (x,y))
                    curIsland=set()
                    while position:
                        cur=heapq.heappop(position)
                        area+=1
                        curIsland.add(cur)
                        for neighbor in neighbors:

                            if cur[0]-neighbor[0]<0 or cur[0]-neighbor[0]>=i: continue
                            if cur[1]-neighbor[1]<0 or cur[1]-neighbor[1]>=j: continue

                            if label[cur[0]-neighbor[0],cur[1]-neighbor[1]]!=0 and (cur[0]-neighbor[0],cur[1]-neighbor[1]) not in seen:
                                seen.add((cur[0]-neighbor[0],cur[1]-neighbor[1]))
                                heapq.heappush(position, (cur[0]-neighbor[0],cur[1]-neighbor[1]))
                
                    maxArea=area
                    maxPos=curIsland
                        
    return island, maxArea, maxPos

def changeClassResult(segmentation):

    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            for z in range(segmentation.shape[2]):
                if segmentation[x,y,z]==3:
                    segmentation[x,y,z]=4
                #CSF into class10
                elif segmentation[x,y,z]==1:
                    segmentation[x,y,z]=10
def changeClassGT(segmentation):
    for x in range(segmentation.shape[0]):
        for y in range(segmentation.shape[1]):
            for z in range(segmentation.shape[2]):
                
                if segmentation[x,y,z]==5:
                    segmentation[x,y,z]=2
                
                elif segmentation[x,y,z]==3 or segmentation[x,y,z]==6:
                    segmentation[x,y,z]=1
    
                    
                    
def saveImage(array, name):
    img = nib.Nifti1Image(array, np.eye(4))
    nib.save(img, name)  

def segVent(imgName, outputPath, resultName):
    result=nib.load(os.path.join(outputPath, resultName)).get_fdata()

    x,y,z=result.shape

    changeClassResult(result)

    #step 1: get subarachnoid connected to skull
    connectToBoundary(result, 10, tolerance=5)

    #step 2: get max area of remaining CSF

    island, Area, maxPos=maxArea(result, 10)
    for pos in island:
        result[pos]=1
    # saveImage(result, 'vent'+resultName)

    for k in range(maxPos[2]-3,maxPos[2]+4):

        if k==maxPos[2]:continue
        for i in range(x):
            for j in range(y):
                if result[i,j,k]==10 and result[i,j,maxPos[2]]==1:
                    result[i,j,k]=1

        Connectivity(result[:,:,k], 10, 1, refClass=1)

    for k in range(z):
        for i in range(x):
            for j in range(y):
                if result[i,j,k]==10:
                    result[i,j,k]=3


    saveImage(result[maxPos[2]-3:maxPos[2]+4], os.path.join(outputPath, 'vent'+resultName))
    
    return Area, maxPos
    

    #Step 3: seg cerebellum
    # for k in range(z//2, -1, -1):
    #     if result[:,:,k].any():
    #         island, maxArea, maxPos=numIsland(result[:,:,k],connectivity=8)
    #         if island>1:
    #             break
    # # print(imageName,k, island, maxArea)            

    # startCere=k
    # for pos in maxPos:
    #     if result[pos[0],pos[1],k]==2:
    #         result[pos[0],pos[1],k]=5


    # for k2 in range(startCere+1, -1, -1):
    #     for i in range(x):
    #         for j in range(y):
    #             if result[i,j,k2]==2 and (i,j) in maxPos:
    #                 result[i,j,k2]=5

    # saveImage(result, 'cere_'+resultName)







