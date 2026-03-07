from collections import Counter
from pathlib import Path
import zlib

import cv2
import numpy as np
import onnxruntime as ort


class YoloOnnxDetector:
	def __init__(
		self,
		onnx_model_path: Path,
		class_names_path: Path,
		confidence: float,
		nms_threshold: float,
		model_size: int,
	):
		self.confidence = confidence
		self.nms_threshold = nms_threshold
		self.model_size = model_size

		if not onnx_model_path.exists():
			raise FileNotFoundError(
				f'ONNX model not found: {onnx_model_path}. Export/download yolov8n.onnx first.'
			)

		with open(class_names_path, 'rt') as file:
			self.class_names = file.read().rstrip('\n').split('\n')

		self.session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
		model_input = self.session.get_inputs()[0]
		self.input_name = model_input.name

		input_shape = model_input.shape
		input_height = input_shape[2] if len(input_shape) > 2 else model_size
		input_width = input_shape[3] if len(input_shape) > 3 else model_size

		self.input_height = int(input_height) if isinstance(input_height, int) else model_size
		self.input_width = int(input_width) if isinstance(input_width, int) else model_size

	def infer(self, frame):
		image = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype(np.float32) / 255.0
		image = np.transpose(image, (2, 0, 1))
		input_tensor = np.expand_dims(image, axis=0)
		return self.session.run(None, {self.input_name: input_tensor})

	def postprocess(self, outputs, frame):
		frame_height, frame_width, _ = frame.shape
		predictions = outputs[0]

		if predictions.ndim == 3:
			predictions = predictions[0]
			if predictions.shape[0] < predictions.shape[1]:
				predictions = predictions.transpose(1, 0)
		elif predictions.ndim != 2:
			return []

		if predictions.shape[1] < 6:
			return []

		num_classes = predictions.shape[1] - 4
		if num_classes <= 0:
			return []

		scale_x = frame_width / float(self.input_width)
		scale_y = frame_height / float(self.input_height)

		boxes = []
		class_ids = []
		confidences = []

		for row in predictions:
			class_scores = row[4:]
			class_id = int(np.argmax(class_scores))
			confidence = float(class_scores[class_id])
			if confidence < self.confidence:
				continue

			center_x, center_y, width, height = row[:4]
			x = int((center_x - width / 2.0) * scale_x)
			y = int((center_y - height / 2.0) * scale_y)
			width = int(width * scale_x)
			height = int(height * scale_y)

			x = max(0, x)
			y = max(0, y)
			width = max(1, min(width, frame_width - x))
			height = max(1, min(height, frame_height - y))

			boxes.append([x, y, width, height])
			class_ids.append(class_id)
			confidences.append(confidence)

		if not boxes:
			return []

		detections = []
		indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
		if len(indices) == 0:
			return detections

		for idx in np.array(indices).flatten():
			x, y, width, height = boxes[idx]
			class_id = class_ids[idx]
			class_name = self.class_names[class_id].upper() if class_id < len(self.class_names) else f'CLASS_{class_id}'
			score = confidences[idx]
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
			color = YoloOnnxDetector._color_for_class(class_name)
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
