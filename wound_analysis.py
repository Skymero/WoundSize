from glob import glob
import pathlib
import cv2
from deepskin import wound_segmentation
from deepskin import evaluate_PWAT_score
import matplotlib.pyplot as plt
import numpy as np
import sys 
#import Detection_algo.CoinDetection as CoinDetection


import time

# Constants
SPEED_OF_LIGHT = 3e8  # Speed of light in meters/second

def measure_reflection_time():
    """
    This function simulates the measurement of the time taken for the IR signal
    to return to the sensor. In a real system, you would replace this with actual
    hardware code interfacing with the IR sensor.
    """
    # Emit IR pulse (in a real project, you'd trigger this with the hardware)
    print("Emitting IR pulse...")
    
    # Simulate time delay (in real code, you'd wait for the actual return signal)
    start_time = time.time()  # Record the start time of the emitted pulse
    # Wait for reflection (simulated delay)
    time.sleep(0.000001)  # Simulated delay for the signal return (1 microsecond)
    end_time = time.time()  # Record the end time when the reflection is detected

    # Calculate the round trip time
    round_trip_time = end_time - start_time
    return round_trip_time

def calculate_distance(time_of_flight):
    """
    Calculate the distance based on the time of flight (time taken for light to travel 
    to the object and reflect back to the sensor).
    """
    # Calculate distance (in meters)
    distance = (SPEED_OF_LIGHT * time_of_flight) / 2
    return distance

# # Simulate the time of flight measurement from the IR sensor
# reflection_time = measure_reflection_time()
# print(f"Time of flight (seconds): {reflection_time}")

# # Calculate the distance to the object
# distance = calculate_distance(reflection_time)
# print(f"Distance to object: {distance} meters")


def getArea():
    #radius * 2 = objSize
    # 158px = 0.9in
    # 79px = 0.45in
    # px = 0.00569620253164556962025316455696 in
    # px^2 = 3.2446723281525396571062329754847 * pow(10,-5) in^2
    
    # px^2 * pixelCount = pixelArea
    # in^2 * pixelCount = pixelArea

    # objectSize = 0.9
    # pixelPerInch = (objectSize /2) / radius
    areaPerPixel = 3.2446723281525396571062329754847 * pow(10,-5)
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

def main(f):

    # files = glob('C:\\Users\\MartinezR\\AI_Scripts\\woundData\\*.png')
    # files = glob('\\WoundSize\\images\\*.png')
    # current_dir = pathlib.Path(__file__).parent
    # images_dir = current_dir / 'images'
    # files = glob(str(images_dir / '*.png'))

    if f is None:
        print(f"No image found.")
    else:
        print(f"Image found")



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    img = cv2.imread(f)[..., ::-1]
        
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


    # mask the image according to the wound
    wound_masked = cv2.bitwise_and(
        img, 
        img, 
        mask=wound_mask
    )

    if wound_mask is None:
        print('wound_mask is None')
    else:
        print('wound_mask is OK')
        # print(body_mask)
        with open('wound_mask.txt', 'w') as fileB:
            for row in wound_mask:
                fileB.write(' '.join(map(str, row)) + '\n')


    wound_area = getArea() 
    print("Wound area: ", wound_area)

    # display the result
    ax.imshow(wound_masked)
    ax.imshow(img, alpha=0.75)
    ax.contourf(wound_mask, cmap='Reds', alpha=0.5)
    ax.contour(wound_mask, colors='lime', linewidths=1)
    ax.grid()
    
    # ax.add_patch(circle)
    ax.text(0, 0, "", fontsize=20, color='k', weight='bold')
    t1 = ax.text(0.3, 0.2, f"Wound Area:{wound_area:.3f}", 
            transform=ax.transAxes, fontsize=14)
    t = ax.text(0.3, 0.05, f"Deepskin's score: {pwat:.3f}",
                transform=ax.transAxes, fontsize=14)
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    ax.axis('off')
    
    outfile = "output_img.png"
    plt.savefig(outfile, bbox_inches='tight', pad_inches=0,dpi=300)
        
    fig.tight_layout()
    plt.show()

main("images/wound.png")