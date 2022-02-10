class InferenceCalculator2:
    @staticmethod
    def get_cls_results(db_inference, det_results, ground_truths, class_id, image_id=None):
        """Get det results and gt information of a certain class.

        Args:
            predictions dict([list]): dict key represents img_id, value is the list of all annotations under this image
            ground_truths: same as predictions
            class_id (int): ID of a specific class.

        Returns:
            list[ndarray(shape)]: outer list represents image, detected bboxes, gt bboxes
        """
        # check if predictions and ground truth dict share the same image_id keys
        db_data = db_inference.storage_task.data
        images = models.DataImage.objects.filter(data=db_data).values_list('image_id', flat=True)
        cls_dets = []
        cls_gts = []

        for img_id in images:
            if image_id is not None and img_id != image_id:
                continue
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

        ious = InferenceCalculator2.polygon_overlaps(predictions, ground_truth)
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
                preds_iou[i] = max_iou
                tp[0, i] = 1
            else:
                fp[0, i] = 1

        return tp, fp, preds_iou


    @staticmethod
    def eval_map(db_inference, predictions, ground_truths, label_id_map, iou_thr=0.5, image_id=None, nproc=4):
        """
        Evaluate mAP of a dataset

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
        num_imgs = len(ground_truths)
        pool = Pool(nproc)
        eval_results = []

        for c in list(label_id_map.keys())[1:]:  # start from idx 1 becasue idx 0 is the background
            # get all annotations of this class by image
            class_preds, class_gt = InferenceCalculator2.get_cls_results(db_inference, predictions, ground_truths, c, image_id)

            # compute tp and fp for each image with multiple processes
            tpfp = pool.starmap(
                InferenceCalculator2.tpfp_imagenet,
                zip(class_preds, class_gt, [iou_thr for _ in range(num_imgs)]))
            tp, fp, preds_iou = tuple(zip(*tpfp))

            # calculate how many ground_truth of this class among all images
            num_gts = np.zeros(1, dtype=int)
            for j, bbox in enumerate(class_gt):
                num_gts[0] += len(bbox)

            # sort all det bboxes by score, also sort tp and fp
            # flatten 2D ground truth array for convenience, and sort by prediction shape confidence
            flattened_class_preds = []
            for labelshape in class_preds:
                flattened_class_preds.extend(labelshape)

            num_preds = len(flattened_class_preds)
            sort_idx = np.argsort(-np.array([shape.confidence for shape in flattened_class_preds]))
            tp = np.hstack(tp)[:, sort_idx]
            fp = np.hstack(fp)[:, sort_idx]
            label_ious = np.hstack(preds_iou)[:, sort_idx][0]
            confidence = np.array([shape.confidence for shape in flattened_class_preds])[sort_idx]

            if num_preds == 0:
                confidence = np.array([-1])
                label_ious = np.array([-1])
            elif num_gts == 0:
                label_ious = np.array([-1])

            # calculate abs recall and precision values
            abs_tp = np.sum(tp)
            if num_preds == 0 or num_gts[0] == 0:
                abs_recall, abs_precision, abs_f1 = -1, -1, -1
            else:
                abs_recall = abs_tp/num_gts[0]
                abs_precision = abs_tp/num_preds
                if abs_recall == 0 or abs_precision == 0:
                    abs_f1 = 0
                else:
                    abs_f1 = 2*(abs_precision*abs_recall)/(abs_precision+abs_recall)

            # calculate recall and precision with tp and fp
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)[0, :]  # (M,)
            precisions = tp / np.maximum((tp + fp), eps)[0, :]  # (M,)

            eval_results.append({
                'num_gts': num_gts,
                'num_dets': num_preds,
                'abs_recall': abs_recall,
                'abs_precision': abs_precision,
                'abs_f1': abs_f1,
                'recall_arr': recalls,
                'precision_arr': precisions,
                'iou': label_ious,
                'conf_scores': confidence,
            })
        pool.close()

        assert len(eval_results) == len(label_id_map) - 1  # minus 1 because of background

        return eval_results