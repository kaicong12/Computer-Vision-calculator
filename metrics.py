import numpy as np
from multiprocessing import Pool


class InferenceCalculator:
    @staticmethod
    def get_cls_results(det_results, ground_truths, class_id, image_list):
        """Get det results and gt information of a certain class.

        Args:
            predictions dict([list]): dict key represents img_id, value is the list of all annotations under this image
            ground_truths: same as predictions
            class_id (int): ID of a specific class.
            image_list = list of image to query annotations from

        Returns:
            list[ndarray(shape)]: outer list represents image, detected bboxes, gt bboxes
        """
        # check if predictions and ground truth dict share the same image_id keys
        cls_dets = []
        cls_gts = []
        for img_id in image_list:
            tmp_detections = []
            tmp_gt = []
            img_detections = det_results.get(img_id, [])
            img_ground_truth = ground_truths.get(img_id, [])

            for shape in img_detections:
                if shape.label_id == class_id:
                    tmp_detections.append(shape)
            cls_dets.append(tmp_detections)

            for shape in img_ground_truth:
                if shape.label_id == class_id:
                    tmp_gt.append(shape)
            cls_gts.append(tmp_gt)

        # make sure the all images under the inference is captured
        assert len(cls_gts) == len(cls_dets)

        return np.array(cls_dets, dtype='object'), np.array(cls_gts, dtype='object')  # make it into ndarray for hstack

    @staticmethod
    def polygon_overlaps(polygon1, polygon2, eps=1e-6):
        """Calculate the ious between each bbox of bboxes1 and bboxes2.
        All bboxes1 and bboxes2 are of the same label and same image_id

        Args:
            polygon1 (ndarray): Shape (n,) -> predictions
            polygon2 (ndarray): Shape (k,) -> ground_truth

        Returns:
            ious (ndarray): Shape (n, k)
        """
        rows = len(polygon1)
        cols = len(polygon2)
        ious = np.zeros((rows, cols), dtype=np.float32)
        if rows * cols == 0:
            return ious

        for idx, polygon in enumerate(polygon1):
            polygon1_shape = polygon.to_polygon()
            for idx2, polygon_2 in enumerate(polygon2):
                polygon2_shape = polygon_2.to_polygon()
                intersect_area = polygon1_shape.intersection(polygon2_shape).area
                union_area = polygon1_shape.union(polygon2_shape).area
                union_area = np.maximum(union_area, eps)
                iou = intersect_area/union_area

                assert iou >= 0

                ious[idx, idx2] = iou

        return ious

    @staticmethod
    def tpfp_imagenet(predictions, ground_truth, default_iou_thr=0.5):
        """
        Check if boxes detected are true positive or false positive
        All boxes here are under the same image

        Returns:
            tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1
        """
        num_preds = len(predictions)
        num_gts = len(ground_truth)
        # tp and fp are of len(num_predictions)
        tp = np.zeros((1, num_preds), dtype=np.float32)
        fp = np.zeros((1, num_preds), dtype=np.float32)
        preds_iou = np.zeros((1, num_preds))
        if len(ground_truth) == 0:
            fp[...] = 1
            return tp, fp, preds_iou

        ious = InferenceCalculator.polygon_overlaps(predictions, ground_truth)
        # sort all detections by scores in descending order
        sort_idx = np.argsort(-np.array([shape.confidence for shape in predictions]))

        gt_covered = np.zeros(num_gts, dtype='bool')
        for i in sort_idx:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlapped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= default_iou_thr and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j

            # there are 2 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 3. it matches no gt, tp = 0, fp = 1
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                preds_iou[0, i] = max_iou
                tp[0, i] = 1
            else:
                fp[0, i] = 1
        return tp, fp, preds_iou

    @staticmethod
    def compute_metric(prediction_arr, gt_arr, tp, fp, iou, mode):
        """Evaluate metric on image level, all annotations here belong to the same image"""
        num_preds, num_gts = len(prediction_arr), len(gt_arr)
        sort_idx = np.argsort(-np.array([shape.confidence for shape in prediction_arr]))
        curr_tp = np.array(tp)[:, sort_idx]
        curr_fp = np.array(fp)[:, sort_idx]
        curr_iou = np.array(iou)[:, sort_idx][0]
        curr_conf = np.array([shape.confidence for shape in prediction_arr])[sort_idx]

        if num_preds == 0 or num_gts == 0:  # if one metric is -1, everything should be -1
            curr_conf = np.array([-1])
            curr_iou = np.array([-1])

        # calculate abs recall and precision values
        abs_tp = np.sum(curr_tp)
        if num_preds == 0 or num_gts == 0:
            abs_recall, abs_precision, abs_f1 = -1, -1, -1
        elif num_preds == 0:
            abs_precision, abs_recall, abs_f1 = -1, 0, -1
        elif num_gts == 0:
            abs_precision, abs_recall, abs_f1 = 0, -1, -1
        else:
            abs_recall = abs_tp / num_gts
            abs_precision = abs_tp / num_preds
            if abs_recall == 0 or abs_precision == 0:
                abs_f1 = 0
            else:
                abs_f1 = 2 * (abs_precision * abs_recall) / (abs_precision + abs_recall)

        curr_tp = np.cumsum(curr_tp, axis=1)
        curr_fp = np.cumsum(curr_fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = curr_tp / np.maximum(np.array([num_gts])[:, np.newaxis], eps)[0, :]  # (M,)
        precisions = curr_tp / np.maximum((curr_tp + curr_fp), eps)[0, :]  # (M,)

        if mode == 'image':
            data = {
                'abs_recall': abs_recall,
                'abs_precision': abs_precision,
                'abs_f1': abs_f1,
                'iou': curr_iou,
                'conf_scores': curr_conf,
            }
        elif mode == 'inference':
            data = {
                'num_gts': num_gts,
                'num_dets': num_preds,
                'abs_recall': abs_recall,
                'abs_precision': abs_precision,
                'abs_f1': abs_f1,
                'recall_arr': recalls,
                'precision_arr': precisions,
                'iou': curr_iou,
                'conf_scores': curr_conf,
            }

        return data

    @staticmethod
    def compute_inference_metric(image_list, image_metrics,
                                   predictions, ground_truth, tp, fp, ious):
        """
        Evaluate metrics for entire inference and on individual image level
        All annotations here are of the same class

        Args:
            image_list [list] -> an array which shows img_id in the same sequence as annotations
            predictions list([list]) -> outer list indicate image, inner list contains predictions under this image
            ground_truth -> format same as predictions, but inner list contains gt
            tp, fp -> same as above
        """
        assert len(predictions) == len(ground_truth) == len(tp) == len(fp) == len(ious)
        curr_idx = 0  # since dictionary does not return key in sequence, identify current image using idx
        # compute metric for this label for individual image
        while curr_idx < len(predictions):
            # curr_<metric> here refers to the metric under current image
            curr_preds = predictions[curr_idx]
            curr_gt = ground_truth[curr_idx]
            curr_tp = tp[curr_idx]
            curr_fp = fp[curr_idx]
            curr_iou = ious[curr_idx]
            image_label_eval_results = InferenceCalculator.compute_metric(
                curr_preds, curr_gt, curr_tp, curr_fp, curr_iou, mode='image'
            )

            curr_image = image_list[curr_idx]
            if curr_image not in image_metrics:
                image_metrics[curr_image] = [image_label_eval_results]
            else:
                image_metrics[curr_image].append(image_label_eval_results)

            curr_idx += 1

        # sort all det bboxes by score, also sort tp and fp
        # flatten 2D ground truth array for convenience, and sort by prediction shape confidence
        flattened_class_preds = []
        flattened_class_gt = []
        for pred_shape, gt_shape in zip(predictions, ground_truth):
            flattened_class_preds.extend(pred_shape)
            flattened_class_gt.extend(gt_shape)

        inference_tp = np.hstack(tp)
        inference_fp = np.hstack(fp)
        inference_ious = np.hstack(ious)
        inference_label_eval_results = InferenceCalculator.compute_metric(
            flattened_class_preds, flattened_class_gt,
            inference_tp, inference_fp, inference_ious, mode='inference'
        )

        return inference_label_eval_results


    @staticmethod
    def eval_map(predictions, ground_truths, label_id_map, image_list, iou_thr, nproc=4):
        """
        Evaluate mAP of a dataset
        Args:
            predictions dict([list]): dict key represents img_id, value is the list of all annotations under this image
            ground_truths: same as predictions
            label_id_map, dict(int): same label_id_map to be reused in plot_pr_curve
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
            mode (str): 'area' or '11points', 'area' means calculating the area
                under precision-recall curve, '11points' means calculating
                the average precision of recalls at [0, 0.1, ..., 1]

        Returns:
            tuple: ([dict, dict, ...]) each dict represents the metrics for the specific label across all images
        """
        num_imgs = len(image_list)
        pool = Pool(nproc)
        eval_results = []
        image_level_metrics = {}

        label_list = sorted(label_id_map.keys())[1:]
        for c in label_list:  # start from idx 1 because idx 0 is the background
            # get all annotations of this class by image
            class_preds, class_gt = InferenceCalculator.get_cls_results(predictions, ground_truths, c, image_list)

            # compute tp and fp for each image with multiple processes
            tpfp = pool.starmap(
                InferenceCalculator.tpfp_imagenet,
                tuple(zip(class_preds, class_gt, [iou_thr for _ in range(num_imgs)])))
            tp, fp, preds_iou = tuple(zip(*tpfp))

            # compute image level and inference level metrics in one function to avoid repeated calling
            inference_label_eval_results = InferenceCalculator.compute_inference_metric(
                image_list, image_level_metrics,
                class_preds, class_gt, tp, fp, preds_iou
            )

            eval_results.append(inference_label_eval_results)
        pool.close()
        assert len(eval_results) == len(label_id_map) - 1  # minus 1 because of background

        return eval_results, image_level_metrics