import time

import cv2

try:
	from ultralytics import YOLO
except Exception:  # pragma: no cover
	YOLO = None


class FallDetector:
	def __init__(self, model_path: str = 'yolov8n-pose.pt', confidence: float = 0.35):
		if YOLO is None:
			raise ImportError('ultralytics YOLO pose backend is unavailable.')

		self.model = YOLO(model_path)
		self.confidence = confidence
		self._last_alert_ts = 0.0

	def detect_fall(self, frame):
		results = self.model.predict(source=frame, conf=self.confidence, imgsz=640, verbose=False, device='cpu')
		if not results:
			return False, None

		result = results[0]
		if result.boxes is None or len(result.boxes) == 0:
			return False, None

		for box in result.boxes.xyxy.cpu().numpy():
			x1, y1, x2, y2 = box
			width = max(1.0, x2 - x1)
			height = max(1.0, y2 - y1)
			ratio = width / height
			center_y = (y1 + y2) * 0.5

			# Heuristic: a mostly horizontal person near lower part of frame may indicate a fall.
			if ratio > 1.2 and center_y > frame.shape[0] * 0.55:
				return True, (int(x1), int(y1), int(width), int(height))

		return False, None

	def should_alert(self, cooldown_seconds: float = 5.0):
		now = time.time()
		if now - self._last_alert_ts < cooldown_seconds:
			return False
		self._last_alert_ts = now
		return True

	@staticmethod
	def draw_alert(frame, box):
		if box is None:
			return
		x, y, width, height = box
		cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)
		cv2.putText(
			frame,
			'FALL RISK',
			(x, max(20, y - 12)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 0, 255),
			2,
		)
