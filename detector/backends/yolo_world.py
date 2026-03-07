from collections import Counter
import zlib

import cv2
import numpy as np

try:
	from ultralytics import YOLOWorld
except Exception:  # pragma: no cover
	YOLOWorld = None


class YoloWorldDetector:
	def __init__(
		self,
		world_model_name: str,
		class_prompts: tuple[str, ...],
		confidence: float,
		nms_threshold: float,
		model_size: int,
	):
		if YOLOWorld is None:
			raise ImportError('ultralytics YOLOWorld backend is unavailable in this environment.')

		self.confidence = confidence
		self.nms_threshold = nms_threshold
		self.model_size = model_size

		self.model = YOLOWorld(world_model_name)
		if class_prompts:
			self.model.set_classes(list(class_prompts))

	def infer(self, frame):
		return self.model.predict(
			source=frame,
			conf=self.confidence,
			iou=self.nms_threshold,
			imgsz=self.model_size,
			verbose=False,
			device='cpu',
		)

	def postprocess(self, outputs, frame):
		if not outputs:
			return []

		result = outputs[0]
		if result.boxes is None or len(result.boxes) == 0:
			return []

		boxes_xyxy = result.boxes.xyxy.cpu().numpy()
		boxes_conf = result.boxes.conf.cpu().numpy()
		boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
		names = result.names

		detections = []
		frame_h, frame_w, _ = frame.shape

		for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
			x = max(0, int(x1))
			y = max(0, int(y1))
			width = max(1, min(int(x2) - x, frame_w - x))
			height = max(1, min(int(y2) - y, frame_h - y))

			class_id = int(boxes_cls[i])
			class_name = str(names.get(class_id, f'class_{class_id}')).upper()
			score = float(boxes_conf[i])
			detections.append((class_name, score, (x, y, width, height)))

		return detections

	@staticmethod
	def _color_for_class(class_name: str):
		seed = zlib.crc32(class_name.encode('utf-8'))
		blue = 80 + (seed & 0x7F)
		green = 80 + ((seed >> 8) & 0x7F)
		red = 80 + ((seed >> 16) & 0x7F)
		return int(blue), int(green), int(red)

	@staticmethod
	def draw_detections(frame, detections):
		for class_name, score, (x, y, width, height) in detections:
			color = YoloWorldDetector._color_for_class(class_name)
			cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
			cv2.putText(
				frame,
				f'{class_name} {int(score * 100)}%',
				(x, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				color,
				2,
			)

	@staticmethod
	def count_by_class(detections):
		counts = Counter()
		for class_name, _, _ in detections:
			counts[class_name] += 1
		return dict(counts)

	@staticmethod
	def draw_count_summary(frame, counts):
		if not counts:
			return

		y = 25
		for class_name in sorted(counts.keys()):
			text = f'{class_name}: {counts[class_name]}'
			cv2.putText(
				frame,
				text,
				(10, y),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 255),
				2,
			)
			y += 28
