import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import Image, ImageDraw

def plot_rectangles(rectangle_array):
    """
    Take in an array of arrays in the format of [x1, y1, x2, y2] ang plot them on a white background

    Args:
    rectangle_array: [ndarray] -> an array of rectangle coordinates
    """

    w, h = 1500, 1500
    # OpenCV method of plotting rectangles on empty image
    # img = np.ones(shape=(w, h, 3), dtype=np.int16)
    # for rect in rectangle_array:
    #     bl = (int(rect[0]), int(rect[1]))
    #     tr = (int(rect[2]), int(rect[3]))
    #     cv2.rectangle(img, bl, tr, color=(255,0,0), thickness=10)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # matplotlib method to create image object and draw rectangle
    # _, ax = plt.subplots()
    # img = np.ones(shape=(w, h, 3), dtype=np.int16)
    # ax.imshow(img)
    # for rect in rectangle_array:
    #     # matplotlib takes in bottom left coordinate as xy, width and height and it accepts float as coordinate
    #     bottom_left = (rect[0], rect[1])
    #     hgt = rect[3] - rect[1]
    #     wdt = rect[2] - rect[0]
    #     rect = patches.Rectangle(bottom_left, wdt, hgt, linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    # plt.show()

    # ImageDraw method to create image object and draw rectangle
    
    img = Image.new("RGB", (w, h))
    img1 = ImageDraw.Draw(img)
    for rect in rectangle_array:
        new_rect = list(map(int, rect))[:-1]  # leave out the confidence score when plotting
        img1.rectangle(new_rect, outline ="red")
    img.show()