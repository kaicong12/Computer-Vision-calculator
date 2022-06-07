# Metrics for Object Detection
## Important Definition
### True Positive, False Positive, False Negative and True Negative  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_  
* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_  
* **False Negative (FN)**: A ground truth not detected  
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding 

### Precision

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFP%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}=\frac{\text{TP}}{\text{all detections}}


### Recall 

Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFN%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}=\frac{\text{TP}}{\text{all ground truths}}


### PR curve (TBC)

### Confusion Matrix (TBC)

### Non-maxima suppression (NMS)
Non Maximum Suppression (NMS) is a technique used in numerous computer vision tasks. It is a class of algorithms to select one entity (e.g., bounding boxes) out of many overlapping entities.
#### Step 1 : 
Select the prediction S with highest confidence score and remove it from P and add it to the final prediction list keep. (keep is empty initially).
#### Step 2 : 
Now compare this prediction S with all the predictions present in P. Calculate the IoU of this prediction S with every other predictions in P. If the IoU is greater than the threshold thresh_iou for any prediction T present in P, remove prediction T from P.
#### Step 3 : 
If there are still predictions left in P, then go to Step 1 again, else return the list keep containing the filtered predictions.