from evaluate_detector import score_predictions


def test_evaluator_computes_precision_recall_and_f1():
    cached = [
        {
            "image": "positive.jpg",
            "ground_truth": [{"box": (10, 10, 50, 50), "class_id": 0}],
            "predictions": [
                {"box": (11, 11, 49, 49), "class_id": 0, "confidence": 0.8},
                {"box": (100, 100, 130, 130), "class_id": 0, "confidence": 0.3},
            ],
        },
        {
            "image": "negative.jpg",
            "ground_truth": [],
            "predictions": [{"box": (5, 5, 20, 20), "class_id": 0, "confidence": 0.7}],
        },
    ]
    report = score_predictions(cached, threshold=0.5, iou_threshold=0.5)
    assert report["true_positives"] == 1
    assert report["false_positives"] == 1
    assert report["false_negatives"] == 0
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0
    assert report["f1"] == 0.6667


def test_evaluator_threshold_can_remove_low_confidence_false_positive():
    cached = [{
        "image": "image.jpg",
        "ground_truth": [{"box": (10, 10, 50, 50), "class_id": 0}],
        "predictions": [
            {"box": (10, 10, 50, 50), "class_id": 0, "confidence": 0.9},
            {"box": (100, 100, 130, 130), "class_id": 0, "confidence": 0.2},
        ],
    }]
    low = score_predictions(cached, threshold=0.1, iou_threshold=0.5)
    high = score_predictions(cached, threshold=0.5, iou_threshold=0.5)
    assert low["false_positives"] == 1
    assert high["false_positives"] == 0
    assert high["f1"] > low["f1"]
