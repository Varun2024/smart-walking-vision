def box_iou(box_a, box_b):
	ax, ay, aw, ah = box_a
	bx, by, bw, bh = box_b

	a_x2 = ax + aw
	a_y2 = ay + ah
	b_x2 = bx + bw
	b_y2 = by + bh

	inter_x1 = max(ax, bx)
	inter_y1 = max(ay, by)
	inter_x2 = min(a_x2, b_x2)
	inter_y2 = min(a_y2, b_y2)

	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h
	if inter_area == 0:
		return 0.0

	area_a = aw * ah
	area_b = bw * bh
	union = max(1, area_a + area_b - inter_area)
	return inter_area / union


def merge_detections(primary, secondary, iou_threshold=0.5):
	merged = list(primary)
	for sec_name, sec_score, sec_box in secondary:
		duplicate = False
		for pri_name, _, pri_box in primary:
			if pri_name == sec_name and box_iou(pri_box, sec_box) >= iou_threshold:
				duplicate = True
				break
		if not duplicate:
			merged.append((sec_name, sec_score, sec_box))
	return merged
