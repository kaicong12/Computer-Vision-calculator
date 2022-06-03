from tests.shapes import *
import numpy as np
from utils import plot_rectangles

def nms(rectangle_array, iou_threshold=0.5):
    """
    Perform NMS on all bounding boxes in the array and return the most appropriate bounding boxes
    This function assumes that all bbox are of the same class
    
    Args:
        rectangle_array (Shape: [, 5]):
            A 2D array with the inner array having [x1 y1 x2 y2 c] format. x1y1 being the bottom left coordinate, x2y2 top right and c is the confidence scores

    Returns:
        Filtered array with of most appropriate bounding boxes
       
    """

    # we extract coordinates for every prediction box present in rectangle_array
    x1 = rectangle_array[:, 0]
    y1 = rectangle_array[:, 1]
    x2 = rectangle_array[:, 2]
    y2 = rectangle_array[:, 3]

    scores = rectangle_array[:, 4]

    # 1. sort the confidence score from low to high
    order = scores.argsort()
    area = (x2 - x1) * (y2 - y1)
    
    keep = []
    while len(order) > 0:
        # 2. Remove the bbox with the highest confidence score from the input first and append it into the "keep" array
        highest_conf_idx = order[-1]
        keep.append(rectangle_array[highest_conf_idx])
        order = order[:-1]

        # 3. Filter out remaining boxes
        xx1 = x1[order]
        xx2 = x2[order]
        yy1 = y1[order]
        yy2 = y2[order]

        # 4. Calculate IoU of each bbox against the removed bbox
        # Maximum gets the top right corner of the intersection bbox
        interx1 = np.maximum(xx1, int(x1[highest_conf_idx]))
        intery1 = np.maximum(yy1, int(y1[highest_conf_idx]))

        # Minimum gets the bottom left corner of the intersection bbox
        interx2 = np.minimum(xx2, int(x2[highest_conf_idx]))
        intery2 = np.minimum(yy2, int(y2[highest_conf_idx]))
        
        w = (interx2 - interx1)
        h = (intery2 - intery1)
        w = np.maximum(w, 1e-6)  # to prevent negative or 0 width and height
        h = np.maximum(h, 1e-6)
        inter_area = w * h

        # Since inter_area is sorted and filtered, sort and filter area to calculate union area
        rem_area = area[order]
        union_area = (rem_area + area[highest_conf_idx]) - inter_area

        IoU = inter_area/union_area
        
        # 5. Remove (suppress) bbox which are above iou_threhsold
        mask = IoU < iou_threshold
        order = order[mask]

    return keep
        
arr = np.array([rect_1, rect_2, rect_3, rect_4, rect_5, rect_6])
keep = nms(arr, 0.99)
plot_rectangles(keep)