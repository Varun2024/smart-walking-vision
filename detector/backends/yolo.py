from pathlib import Path
from collections import Counter
import zlib

import cv2
import numpy as np


class YoloDetector:
	def __init__(
		self,
		cfg_path: Path,
		weights_path: Path,
		class_names_path: Path,
		confidence: float,
		nms_threshold: float,
		model_size: int,
	):
		self.confidence = confidence
		self.nms_threshold = nms_threshold
		self.model_size = model_size

		with open(class_names_path, 'rt') as file:
			self.class_names = file.read().rstrip('\n').split('\n')

		self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

		layer_names = self.net.getLayerNames()
		self.output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

	def infer(self, frame):
		blob = cv2.dnn.blobFromImage(
			frame,
			1 / 255,
			(self.model_size, self.model_size),
			[0, 0, 0],
			1,
			crop=False,
		)
		self.net.setInput(blob)
		return self.net.forward(self.output_names)

	def postprocess(self, outputs, frame):
		frame_height, frame_width, _ = frame.shape

		boxes = []
		class_ids = []
		confidences = []

		for output in outputs:
			for detection in output:
				scores = detection[5:]
				class_id = int(np.argmax(scores))
				confidence = float(scores[class_id])
				if confidence <= self.confidence:
					continue

				width = int(detection[2] * frame_width)
				height = int(detection[3] * frame_height)
				x = int((detection[0] * frame_width) - width / 2)
				y = int((detection[1] * frame_height) - height / 2)

				boxes.append([x, y, width, height])
				class_ids.append(class_id)
				confidences.append(confidence)

		detections = []
		indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
		if len(indices) == 0:
			return detections

		for idx in np.array(indices).flatten():
			x, y, width, height = boxes[idx]
			class_name = self.class_names[class_ids[idx]].upper()
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
			color = YoloDetector._color_for_class(class_name)
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
