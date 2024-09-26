from glob import glob
import cv2
from deepskin import wound_segmentation
from deepskin import evaluate_PWAT_score
import matplotlib.pyplot as plt
import numpy as np
import sys 
#import Detection_algo.CoinDetection as CoinDetection

def getArea(radius):
    #radius * 2 = objSize
    # 158px = 0.9in
    # 79px = 0.45in
    # px = 0.00569620253164556962025316455696 in
    # px^2 = 3.2446723281525396571062329754847 * pow(10,-5) in^2
    
    # px^2 * pixelCount = pixelArea
    # in^2 * pixelCount = pixelArea

    objectSize = 0.9
    pixelPerInch = (objectSize /2) / radius
    areaPerPixel = pixelPerInch * pixelPerInch
    pixelCount = getPixelCount()
    woundArea = areaPerPixel * pixelCount 
    
    return woundArea

def getPixelCount():
    totalPixelCount = 0
    # rowCount = 0
    # colCount = 0
    # colCountArray = []
    with open('wound_mask.txt', 'r') as f:      
        for line in f:
            # if '255' in line:
            #     rowCount += 1
            line = line.split()
            #print(line)
            for i in line:
                # if i == '255':
                #     colCount += 1
                    # colCountArray.append(colCount)
                if i == '255' or i[0] > '0': #and rowCount > 261:
                    # if count == 0:
                    #     print(i)
                    totalPixelCount += 1
            # colCount = 0
        # maxColCount = max(colCountArray)
    # print("How may pixel columns: ", maxColCount)
    # print("How many rows: ", rowCount)
    print("Total pixel count: ", totalPixelCount)

    return totalPixelCount

files = glob('C:\\Users\\MartinezR\\AI_Scripts\\woundData\\wound10.png')

if not files:
    print("No files found.")
else:
    print(f"Files found: {files}")

label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 21))

for f, ax, lbl in zip(files, axes.ravel(), label):
    
    img = cv2.imread(f)[..., ::-1]
    
    if img is None:
        print(f"Failed to load image: {f}")
        continue
    
    # # load the image in RGB fmt
    # img = cv2.imread(f)[..., ::-1]
    
    # get the semantic segmentation mask
    mask = wound_segmentation(
      img=img,
      tol=0.5,
      verbose=False,
    )

    if mask is None:
        print(f"Failed to get segmentation mask for image: {f}")
        continue

    # un-pack the semantic mask
    wound_mask, body_mask, bg_mask = cv2.split(mask)
    
    # compute the wound PWAT
    pwat = evaluate_PWAT_score(
      img=img,
      mask=mask,
      verbose=False,
    )


    if pwat is None:
        print(f"Failed to evaluate PWAT score for image: {f}")
        continue

    # mask the image according to the wound
    wound_masked = cv2.bitwise_and(
        img, 
        img, 
        mask=wound_mask
    )


    # if body_mask is None:
    #     print('body_mask is None')
    # else:
    #     print('body_mask is OK')
    #     # print(body_mask)
    #     with open('body_mask.txt', 'w') as fileB:
    #         for row in body_mask:
    #             fileB.write(' '.join(map(str, row)) + '\n')


    if wound_mask is None:
        print('wound_mask is None')
    else:
        print('wound_mask is OK')
        # print(body_mask)
        with open('wound_mask.txt', 'w') as fileB:
            for row in wound_mask:
                fileB.write(' '.join(map(str, row)) + '\n')

    ################################################################
    # print(type(f))
    # # image_path = f.read()
    # # print(type(f))
    # x, y, r = CoinDetection.CoinDetection(f)
    # print("x, y, r: ", x, y, r)
    r = 79
    
    # circle = plt.Circle((x, y), r, color='r', fill=False)

    #pixelArea, pxlPerMetric = getPixerPerMetric(r) #(in^2 / px , px/in)

    #################################################################
    wound_area = getArea(r) 
    print("Wound area", lbl,": ", wound_area)

    # display the result
    ax.imshow(wound_masked)
    ax.imshow(img, alpha=0.75)
    ax.contour(wound_mask, colors='lime', linewidths=1)
    ax.grid()
    
    # ax.add_patch(circle)
    ax.text(-50, -20, lbl, fontsize=40, color='k', weight='bold')
    t1 = ax.text(0.3, 0.2, f"Wound Area:{wound_area:.3f}", 
             transform=ax.transAxes, fontsize=14)
    t = ax.text(0.3, 0.05, f"Deepskin's score: {pwat:.3f}",
                transform=ax.transAxes, fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    ax.axis('off')
    
fig.tight_layout()
plt.show()