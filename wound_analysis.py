from glob import glob
import cv2
from deepskin.segmentation import wound_segmentation
from deepskin.pwat import evaluate_PWAT_score
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


def getArea(radius):
    #radius * 2 = objSize
    # 158px = 0.9in -> the full width|diameter of the quarter is 158 pixels in length
    # 175.556 px  = 1 in
    # 79px = 0.45in -> quarter's radius
    # pixel size lenght = px width = 1/175.556
    # area per pixel = (1/175.556)^2
    
    # px = 0.00569620253164556962025316455696 in 
    # px^2 = 3.2446723281525396571062329754847 * pow(10,-5) in^2
    
    # px^2 * pixelCount = pixelArea_Px
    # in^2 * pixelCount = pixelArea_In
    
    #1 get the size of a pixel - the radius of the object is hardcoded for now
    # TODO: add logic to detect quarter and calculate pixel area per pixel from that
    # we're assuming each wound is at the same distance every time
    
    #objectSize = size of a quarter on inches

    """
    Calculate the area of a given wound image in square inches.

    The area calculation is based on the assumption that the size of the quarter
    is 0.9 inches in diameter and 158 pixels in length. The area per pixel is
    calculated as (1/175.556)^2 square inches. The total area is the product of
    the area per pixel and the total number of pixels in the wound image.

    Returns
    -------
    woundArea : float
        The calculated area of the wound in square inches.
    """

    objectSize = 0.9
    
    pixelPerInch = 158/objectSize # how many pixels equal an inch | pixel/inch
    
    #areaPerPixel = pixelPerInch * pixelPerInch # (px/inch)^2
    areaPerPixel = 3.2446723281525396571062329754847 * pow(10,-5) # (px/inch)^2
    
    
    pixelCount = getPixelCount() # total number of pixels
    
    woundArea = areaPerPixel * pixelCount # (px/inch)^2 * total number of pixels(scalar) -> 
    
    return woundArea

def getPixelCount():

    """
    Count the total number of pixels in the wound mask image.

    Opens the file "wound_mask.txt" and reads it line by line. For each line, it
    splits the line into individual elements and checks if the element is '255'
    or starts with a number greater than '0'. If the element meets one of these
    conditions, it increments the totalPixelCount by 1.

    After processing all the lines in the file, it prints the totalPixelCount
    and returns it.

    Returns
    -------
    int
        The total number of pixels in the wound mask image.
    """

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

def main(image):

    lbl = "wound"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(21, 21))        
    
    # load the image in RGB fmt
    img = cv2.imread(image)[..., ::-1]
    
    if img is None:
        print("Failed to load image")
        return None
        
    # get the semantic segmentation mask
    mask = wound_segmentation(
    img=img,
    tol=0.5,
    verbose=False,
    )

    if mask is None:
        print(f"Failed to get segmentation mask for image")
        return None

    # un-pack the semantic mask
    wound_mask, body_mask, bg_mask = cv2.split(mask)
    
    # compute the wound PWAT
    pwat = evaluate_PWAT_score(
    img=img,
    mask=mask,
    verbose=False,
    )


    if pwat is None:
        print(f"Failed to evaluate PWAT score for image")
        return None

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
        # 
        with open('wound_mask.txt', 'w') as fileB:
            for row in wound_mask:
                fileB.write(' '.join(map(str, row)) + '\n')

    #################################################################
    
    r = 79
    #################################################################
    wound_area = getArea(r) 
    print("Wound area: ", wound_area)

    # display the result
    ax.imshow(wound_masked)
    ax.imshow(img, alpha=0.75)
    ax.contour(wound_mask, colors='lime', linewidths=1)
    # plt.savefig(f'{lbl}.png')
    ax.grid()
    
    # ax.add_patch(circle)
    ax.text(0, 0, lbl, fontsize=8, color='k', weight='bold')
    t1 = ax.text(0.3, 0.2, f"Wound Area:{wound_area:.3f}", 
            transform=ax.transAxes, fontsize=8)
    t = ax.text(0.3, 0.05, f"Deepskin's score: {pwat:.3f}",
                transform=ax.transAxes, fontsize=8)
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
    ax.axis('off')


    
    fig.tight_layout()
    plt.savefig('wound_table.png') 
    returnImage =cv2.imread('wound_table.png')
    returnImage = cv2.resize(returnImage, (800, 600)) 
    # plt.show()

    return 

if __name__ == "__main__":
    filepath = "wound.png"
    p, w, rimg = main(filepath)
    print("Deepskin's score: ", p)
    print("Wound Area: ", w)
    if rimg is not None: 
        print("Displaying image...")
        # cv2.namedWindow('wound', cv2.WINDOW_NORMAL)
        cv2.imshow('Wound Segmented Image', rimg)
    cv2.waitKey(0)
