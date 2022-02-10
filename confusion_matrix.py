def confusion_matrix_per_image(label_id_map, confusion_matrix, ground_truth_arr, preds_arr, iou_threshold):
    """
    When returning labels in viewset make sure the labels are returned in ascending id
    """
    true_positives = np.zeros(len(ground_truth_arr))
    iou_matrix = InferenceCalculator2.polygon_overlaps(preds_arr, ground_truth_arr)

    for pred_idx, pred_shape in enumerate(preds_arr):
        match_ground_truth = 0
        for gt_idx, gt_shape in enumerate(ground_truth_arr):
            if iou_matrix[
                pred_idx, gt_idx] >= iou_threshold and pred_shape.seen == 0:  # one prediction can only belong to one ground truth
                match_ground_truth += 1
                # mark out this pred shape to prevent this from getting reused in the next round
                pred_shape.seen = 1
                gt_shape.seen = 1  # mark out ground truth to help identify FN

                gt_label_id = label_id_map[gt_shape.label_id]
                pred_label_id = label_id_map[pred_shape.label_id]
                pred_shape.coord = [gt_label_id,
                                    pred_label_id]  # store this coord in confusion matrix to be queried later
                confusion_matrix[gt_label_id, pred_label_id] += 1  # model predicts this label

                if gt_shape.label_id == pred_shape.label_id:
                    true_positives[gt_idx] += 1  # allows multiple preds to one gt box (TP)

        # FP -> prediction shape cannot match ground truth
        if match_ground_truth == 0:
            pred_shape.coord = [-1, label_id_map[pred_shape.label_id]]
            confusion_matrix[-1, label_id_map[pred_shape.label_id]] += 1

    gt_labels = [label_id_map[gt.label_id] for gt in ground_truth_arr]
    for num_tp, gt_label, in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN -> model cannot predict ground truth
            confusion_matrix[gt_label, -1] += 1

    for gt_shape, gt_label in zip(ground_truth_arr, gt_labels):
        if gt_shape.seen == 0:
            gt_shape.coord = [gt_label, -1]


def confusion_matrix(ground_truth, predictions, label_id_map, iou_threshold):
    """Calculate confusion matrix for the entire inference"""
    num_classes = len(label_id_map) - 1
    # +1 here is because the last row is referring to 'background' and is used to store FP and FN
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])

    image_list = ground_truth.keys() | predictions.keys()
    for img_id in image_list:
        gt = ground_truth.get(img_id, [])
        preds = predictions.get(img_id, [])
        InferenceCalculator2.confusion_matrix_per_image(
            label_id_map, confusion_matrix,
            gt, preds, iou_threshold
        )

    return confusion_matrix