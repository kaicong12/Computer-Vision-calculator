from .metrics import InferenceCalculator
import numpy as np


class ConfusionMatrix_Evaluator:
    # TODO: merge this function with 'count_tp_fp' and 'get_tp_by_label' to avoid repeated computation
    @staticmethod
    def _iou_matrix(ground_truth_arr, preds_arr):
        """
        Similar function as 'compute_tp_fp' but this returns a
        iou_matrix with shape = (len(gt)+1 x len(preds)+1), used for confusion matrix computation
        The +1 is to store all the FP and FN as background detections
        """
        rows = len(ground_truth_arr)
        cols = len(preds_arr)
        iou_matrix = np.zeros((rows, cols), dtype=np.float32)

        for pred_idx, pred_shape in enumerate(preds_arr):
            pred_s = pred_shape.to_polygon()
            for gt_idx, gt_shape in enumerate(ground_truth_arr):
                gt_s = gt_shape.to_polygon()
                intersect_area = pred_s.intersection(gt_s).area
                union_area = pred_s.union(gt_s).area
                iou = intersect_area / union_area
                iou_matrix[gt_idx, pred_idx] = iou

        return iou_matrix

    @staticmethod
    def confusion_matrix(ground_truth_arr, preds_arr, iou_threshold):
        # when returning labels in viewset, make sure the labels are returned in ascending id
        label_id_map = {}
        label_ids = list(set([gt.label_id for gt in
                              ground_truth_arr]))  # utilize the fact that set returns elements in sorted manner
        counter = 0  # This confusion matrix computation assumes label_id begins with 0
        for pk in label_ids:
            if pk not in label_id_map:
                label_id_map[pk] = counter
                counter += 1

        num_classes = len(label_id_map)
        # +1 here is because the last row is referring to 'background' and is used to store FP and FN
        confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])

        true_positives = np.zeros(len(ground_truth_arr))
        iou_matrix = InferenceCalculator._iou_matrix(ground_truth_arr, preds_arr)

        for pred_idx, pred_shape in enumerate(preds_arr):
            match_ground_truth = 0
            for gt_idx, gt_shape in enumerate(ground_truth_arr):
                if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
                    match_ground_truth += 1
                    gt_label_id = label_id_map[gt_shape.label_id]
                    pred_label_id = label_id_map[pred_shape.label_id]
                    confusion_matrix[gt_label_id, pred_label_id] += 1  # model predicts this label

                    if gt_shape.label_id == pred_shape.label_id:
                        true_positives[gt_idx] += 1  # allows multiple preds to one gt box (TP)

            # FP -> prediction shape cannot match ground truth
            if match_ground_truth == 0:
                confusion_matrix[-1, label_id_map[pred_shape.label_id]] += 1

        gt_labels = [label_id_map[gt.label_id] for gt in ground_truth_arr]
        for num_tp, gt_label in zip(true_positives, gt_labels):
            if num_tp == 0:  # FN -> model cannot predict ground truth
                confusion_matrix[gt_label, -1] += 1

        return confusion_matrix