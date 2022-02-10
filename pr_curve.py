def pr_curve(eval_results, mode, label_id_map, num_points, smooth=True):
    """
    eval_results -> precision and recall by labels
    mode = "avg" -> average precision and recall values over classes
    mode = "wavg" -> weighted average ...
    label_id_map dict(int) -> enumerate of each label_id in the dataset to get the idx of this label in eval_results
    num_points int -> number of threshold on the pr_curve x-axis
    otherwise, mode should be an integer denoting the class index

    Return: precision, recall array for the mode specified
    """
    recalls = [eval_results_i["recall_arr"] for eval_results_i in eval_results]
    precisions = [eval_results_i["precision_arr"] for eval_results_i in eval_results]
    conf_scores = [eval_results_i["conf_scores"] for eval_results_i in eval_results]

    if isinstance(mode, int):
        label_idx = label_id_map.get(mode, None)
        if label_idx is None:
            raise ValidationError('Label ID not mapped to label_id_map')

        recalls = [recalls[label_idx]]
        precisions = [precisions[label_idx]]
        conf_scores = [conf_scores[label_idx]]

    # Get max and min confidence scores
    min_score = 0
    max_score = 0
    for conf_scores_i in conf_scores:
        for score in conf_scores_i:
            min_score = min(score, min_score)
            max_score - max(score, max_score)

    # Calculate precision and recall values over multiple points
    score_range = np.linspace(
        start=min_score, stop=max_score,
        endpoint=True, num=num_points)

    recalls_at_scores = []
    precisions_at_scores = []
    for threshold in score_range:
        recalls_at_score_i = []
        precisions_at_score_i = []
        for recalls_i, precisions_i, conf_scores_i in \
                zip(recalls, precisions, conf_scores):
            if recalls_i.shape[1] == 0:
                # if this label does not have annotation, treat recall and precision as 0
                recalls_at_score_i.append(0)
                precisions_at_score_i.append(0)
            else:
                mask = (conf_scores_i <= threshold)
                if not mask.any():
                    i = len(conf_scores_i) - 1
                else:
                    i = mask.argmax()

                recalls_at_score_i.append(recalls_i[0][i])
                precisions_at_score_i.append(
                    precisions_i[0][i:].max() if smooth
                    else precisions_i[0][i]
                )

            # Aggregate
        if mode in ["avg", "wavg"]:
            if mode == "avg":
                weights = np.array([1] * len(recalls))
            else:
                weights = np.array([
                    eval_results_i["num_gts"]
                    for eval_results_i in eval_results
                ])

            recalls_at_score_i = np.array(recalls_at_score_i)
            precisions_at_score_i = np.array(precisions_at_score_i)

            # Calculate weighted sum
            assert len(recalls_at_score_i) == len(precisions_at_score_i) == len(weights)
            recall_at_score_i = (recalls_at_score_i * weights).sum() / weights.sum()
            precision_at_score_i = (precisions_at_score_i * weights).sum() / weights.sum()

        else:
            assert isinstance(mode, int)
            assert len(recalls_at_score_i) == len(precisions_at_score_i) == 1
            recall_at_score_i = recalls_at_score_i[0]
            precision_at_score_i = precisions_at_score_i[0]

        recalls_at_scores.append(recall_at_score_i)
        precisions_at_scores.append(precision_at_score_i)

    return recalls_at_scores, precisions_at_scores
