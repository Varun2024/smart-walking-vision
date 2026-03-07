from detector.backends.merge_utils import box_iou, merge_detections


def test_box_iou_overlap():
    box_a = (10, 10, 40, 40)
    box_b = (20, 20, 40, 40)
    iou = box_iou(box_a, box_b)
    assert 0 < iou < 1


def test_merge_detections_avoids_duplicate_same_class():
    primary = [('PERSON', 0.9, (10, 10, 40, 40))]
    secondary = [
        ('PERSON', 0.8, (12, 12, 40, 40)),
        ('BOTTLE', 0.7, (100, 100, 20, 40)),
    ]

    merged = merge_detections(primary, secondary, iou_threshold=0.5)

    labels = [name for name, _, _ in merged]
    assert labels.count('PERSON') == 1
    assert labels.count('BOTTLE') == 1
