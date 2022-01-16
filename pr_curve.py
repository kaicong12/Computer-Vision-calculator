from .metrics import InferenceCalculator
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics_per_inference(ground_truth_arr_by_image, preds_arr_by_image, iou_threshold, label_id=None):
    image_list = list(ground_truth_arr_by_image.keys() | preds_arr_by_image.keys())

    predictions_used = []
    overall_acc_tp = []
    overall_acc_fp = []
    # compute acc_tp and acc_fp before combining, so that polygons across images would not interact
    for img_id in image_list:
        gt = ground_truth_arr_by_image.get(img_id, [])
        preds = preds_arr_by_image.get(img_id, [])

        if label_id is not None:
            gt = [g for g in gt if g.label_id == label_id]
            preds = [p for p in preds if p.label_id == label_id]

        tmp_acc_tp, tmp_acc_fp, _ = InferenceCalculator._count_tp_fp(gt, preds, iou_threshold)

        # label each predictions with a fp or tp label
        predictions_used.extend(preds)
        overall_acc_tp.extend(tmp_acc_tp)
        overall_acc_fp.extend(tmp_acc_fp)

    sort_idx = np.argsort([pred_shape.confidence for pred_shape in predictions_used])
    sorted_acc_tp = list(np.array(overall_acc_tp)[sort_idx])
    sorted_acc_fp = list(np.array(overall_acc_fp)[sort_idx])

    tp = np.sum(sorted_acc_tp)
    acc_tp = np.cumsum(sorted_acc_tp)
    acc_fp = np.cumsum(sorted_acc_fp)

    return acc_tp, acc_fp, tp


def plot_pr_curve(ground_truth_by_image, predictions_by_image, iou_threshold, label_id=None):
    """
    Compute PR Curve on overall task level using annotations from each image
    :param ground_truth_by_image: Ground truth annotation dictionary, with image_id as key
    :param predictions_by_image: Predictions dictionary, with image_id as key
    :param iou_threshold: Iou Threshold to compute TP
    :param label_id: If specified, will only compute metrics based on annotation of this label id
    :return:
    """
    acc_tp, acc_fp, tp = InferenceCalculator.compute_metrics_per_inference(
        ground_truth_by_image, predictions_by_image,
        iou_threshold, label_id
    )
    base_precision_arr = acc_tp / (acc_tp + acc_fp)
    ground_truth_len = 0
    for annotations in predictions_by_image.values():
        for anno_shape in annotations:
            if label_id is not None:
                if anno_shape.label_id == label_id:
                    ground_truth_len += 1
            else:
                ground_truth_len += 1
    base_recall_arr = acc_tp / ground_truth_len

    plt.plot(base_recall_arr, base_precision_arr)
    plt.show()
