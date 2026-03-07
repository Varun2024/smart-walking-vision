from collections import defaultdict, deque


def box_center(box):
	x, y, width, height = box
	return x + width * 0.5, y + height * 0.5


class TemporalSmoother:
	def __init__(self, iou_threshold: float = 0.35):
		self.iou_threshold = iou_threshold
		self._tracks = []

	@staticmethod
	def _iou(box_a, box_b):
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
		if inter_area <= 0:
			return 0.0

		area_a = max(1, aw * ah)
		area_b = max(1, bw * bh)
		return inter_area / max(1, area_a + area_b - inter_area)

	def update(self, detections):
		updated_tracks = []
		matched = [False] * len(detections)

		for track in self._tracks:
			best_idx = -1
			best_iou = 0.0
			for idx, (name, score, box) in enumerate(detections):
				if matched[idx] or name != track['name']:
					continue
				iou = self._iou(track['box'], box)
				if iou > best_iou:
					best_iou = iou
					best_idx = idx

			if best_idx >= 0 and best_iou >= self.iou_threshold:
				name, score, box = detections[best_idx]
				prev_cx, prev_cy = box_center(track['box'])
				curr_cx, curr_cy = box_center(box)
				vx = curr_cx - prev_cx
				vy = curr_cy - prev_cy
				updated_tracks.append({'name': name, 'score': score, 'box': box, 'velocity': (vx, vy), 'missed': 0})
				matched[best_idx] = True
			else:
				track['missed'] += 1
				if track['missed'] <= 4:
					updated_tracks.append(track)

		for idx, det in enumerate(detections):
			if matched[idx]:
				continue
			name, score, box = det
			updated_tracks.append({'name': name, 'score': score, 'box': box, 'velocity': (0.0, 0.0), 'missed': 0})

		self._tracks = updated_tracks

	def predict(self):
		predicted = []
		for track in self._tracks:
			x, y, width, height = track['box']
			vx, vy = track['velocity']
			nx = int(x + vx)
			ny = int(y + vy)
			track['box'] = (nx, ny, width, height)
			predicted.append((track['name'], track['score'], track['box']))
		return predicted


class CountPersistence:
	def __init__(self, history_size: int = 5):
		self.history_size = max(1, history_size)
		self._history = defaultdict(lambda: deque(maxlen=self.history_size))

	def smooth(self, current_counts):
		keys = set(self._history.keys()) | set(current_counts.keys())
		for name in keys:
			value = current_counts.get(name, 0)
			self._history[name].append(value)

		smoothed = {}
		to_remove = []
		for name, values in self._history.items():
			avg_value = sum(values) / len(values)
			stable = int(round(avg_value))
			if stable > 0:
				smoothed[name] = stable
			elif all(v == 0 for v in values):
				to_remove.append(name)

		for name in to_remove:
			del self._history[name]

		return smoothed


def estimate_proximity_label(frame_shape, box):
	frame_h, frame_w = frame_shape[:2]
	_, _, width, height = box
	area_ratio = (width * height) / max(1, frame_w * frame_h)

	if area_ratio >= 0.16:
		return 'VERY CLOSE', 3
	if area_ratio >= 0.08:
		return 'CLOSE', 2
	if area_ratio >= 0.03:
		return 'MEDIUM', 1
	return 'FAR', 0
