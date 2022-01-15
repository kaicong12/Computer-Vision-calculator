import numpy as np
from enum import Enum
from shapely import geometry
from shapely.validation import make_valid


def pairwise(iterable):
    a = iter(iterable)
    return list(zip(a, a))


class AnnoType(str, Enum):
    GROUND_TRUTH = 'ground_truth'
    PREDICTIONS = 'predictions'


class InferenceBBox:
    def __init__(self, id, points, label_id, type,
                 seen=0, confidence=None):
        self.id = id
        self.points = points
        self.label_id = label_id
        self.type = type
        self.seen = seen
        self.confidence = confidence

    def to_polygon(self):
        return geometry.box(*self.points)


class InferencePolygon:
    def __init__(self, id, points, label_id, type,
                 seen=0, confidence=None):
        self.id = id
        self.points = points
        self.label_id = label_id
        self.type = type
        self.seen = seen
        self.confidence = confidence

    def to_polygon(self):
        polygon = geometry.Polygon(pairwise(self.points))
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        return polygon

SHAPE_TYPE = {
    'rectangle': InferenceBBox,
    'polygon': InferencePolygon
}


class InferenceCalculator:
    @staticmethod
    def _iou_polygon(polygonA, polygonB):
        if not polygonA.intersects(polygonB):
            return 0

        interArea = polygonA.intersection(polygonB).area
        unionArea = polygonA.union(polygonB).area
        iou = interArea / unionArea

        return iou

    @staticmethod
    def _group_shape_by_label(shape_arr):
        shape_dict = {}
        for s in shape_arr:
            shape = s.to_polygon()

            if s.label_id not in shape_dict:
                shape_dict[s.label_id] = shape
            else:
                # convert into multi-polygon
                shape_dict[s.label_id].union(shape)

        return shape_dict

    @staticmethod
    def _iou(ground_truth_arr, predictions_arr):
        """
        All grd_truth and predictions_arr are under the same image
        """
        # group polygon by label and convert them into multipolygon
        grd_truth = InferenceCalculator._group_shape_by_label(ground_truth_arr)
        preds = InferenceCalculator._group_shape_by_label(predictions_arr)

        # label which does not appear in either grd_truth or preds will score 0
        missing_labels = list(set(grd_truth.keys()) - set(preds.keys()))
        score_by_label_id = {l: 0 for l in missing_labels}
        for label_id in missing_labels:
            if label_id in grd_truth:
                grd_truth.pop(label_id)
            else:
                preds.pop(label_id)

        for label_id, multi_polygon in grd_truth.items():
            p0 = multi_polygon
            p1 = preds[label_id]
            intersect_area = p0.intersection(p1).area
            union_area = p0.union(p1).area
            iou = intersect_area / union_area
            assert iou >= 0

            score_by_label_id[label_id] = iou

        return score_by_label_id

    @staticmethod
    def _count_tp_fp(ground_truth_arr, preds_arr, iou_threshold):
        TP = np.zeros(len(preds_arr))
        FP = np.zeros(len(preds_arr))

        # pairwise comparison between predictions and every ground truth to find bbox with highest iou
        # mark this ground truth as 'seen'
        for p in range(len(preds_arr)):
            iouMax = -1
            seen_gt = -1
            for j in range(len(ground_truth_arr)):
                iou = InferenceCalculator._iou_polygon(preds_arr[p].to_polygon(),
                                                       ground_truth_arr[j].to_polygon())

                if iou > iouMax:
                    iouMax = iou
                    seen_gt = j

            # if iou larger than threshold and this ground truth is not seen yet, mark this prediction as TP
            # used to tackle cases when model predicts multiple BBox on one ground truth
            if iouMax >= iou_threshold:
                if ground_truth_arr[seen_gt].seen == 0:
                    TP[p] = 1
                    ground_truth_arr[seen_gt].seen = 1
                else:
                    # this ground truth has already been used for other predictions
                    FP[p] = 1
            else:
                FP[p] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        TP = np.sum(TP)

        return acc_TP, acc_FP, TP

    @staticmethod
    def _get_tp_by_label(ground_truth_arr, preds_arr, iou_threshold) -> dict():
        """
        takes in 2 arrays of bbox input and
        metrics are computed based on each label_id of the bbox
        """
        classes = []
        # unique label_id in both predictions and ground_truth
        for g in ground_truth_arr:
            if g.label_id not in classes:
                classes.append(g.label_id)

        for p in preds_arr:
            if p.label_id not in classes:
                classes.append(p.label_id)

        # compute tp, fp by class
        tp_by_class = {}
        for c in classes:
            # sort predctions in reverse confidence level to compute PR curve
            preds = [p for p in preds_arr if p.label_id == c]
            predictions = sorted(preds, key=lambda x: x.confidence, reverse=True)
            gt = [grd for grd in ground_truth_arr if grd.label_id == c]
            acc_TP, acc_FP, TP = InferenceCalculator._count_tp_fp(gt, predictions, iou_threshold)

            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            recall = acc_TP/len(gt)
            data =  {
                'TP': TP,
                # precision and recall here are the accumulative precision and recall (array)
                # sorted based on descending prediction confidence
                'PRECISION': prec,
                'RECALL': recall,
            }

            tp_by_class[c] = data

        return tp_by_class

    @staticmethod
    def compute_pr_curve(ground_truth_arr, preds_arr, iou_threshold, label_id):
        metrics_by_label = InferenceCalculator._get_tp_by_label(ground_truth_arr, preds_arr, iou_threshold)
        # precision X recall array at this SPECIFIC 'IoU threshold' and 'confidence threshold'
        precision_arr = metrics_by_label[label_id]['PRECISION']
        recall_arr = metrics_by_label[label_id]['RECALL']

        return precision_arr, recall_arr

    @staticmethod
    def compute_precision(ground_truth_arr, preds_arr, iou_threshold):
        if len(preds_arr) == 0:
            return 0

        tp_by_class = InferenceCalculator._get_tp_by_label(ground_truth_arr, preds_arr, iou_threshold)
        all_tp = sum([tp_by_class[label]['TP'] for label in tp_by_class.keys()])
        precision = all_tp / len(preds_arr)

        return precision

    @staticmethod
    def compute_recall(ground_truth_arr, preds_arr, iou_threshold):
        if len(ground_truth_arr) == 0:
            return 0

        tp_by_class = InferenceCalculator._get_tp_by_label(ground_truth_arr, preds_arr, iou_threshold)
        all_tp = sum([tp_by_class[label]['TP'] for label in tp_by_class])
        recall = all_tp / len(ground_truth_arr)

        return recall

    @staticmethod
    def compute_iou_score_by_label(ground_truth_arr, preds_arr):
        iou_score_by_label = InferenceCalculator._iou(ground_truth_arr, preds_arr)

        # iou score for this image would be the mean iou score for all labels
        return iou_score_by_label


