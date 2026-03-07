import json
from copy import deepcopy
from pathlib import Path


FALLBACK_SETTINGS = {
	'quality_presets': {
		'fast': {'model_size': 320, 'confidence': 0.50, 'nms_threshold': 0.45, 'width': 960, 'height': 540, 'infer_every': 2},
		'balanced': {'model_size': 416, 'confidence': 0.45, 'nms_threshold': 0.40, 'width': 1280, 'height': 720, 'infer_every': 1},
		'accurate': {'model_size': 608, 'confidence': 0.40, 'nms_threshold': 0.40, 'width': 1280, 'height': 720, 'infer_every': 1},
	},
	'world_default_classes': [
		'person', 'watch', 'cell phone', 'backpack', 'bottle', 'chair', 'car', 'traffic cone', 'barrier', 'pothole'
	],
	'defaults': {
		'quality': 'balanced',
		'model_type': 'yolov8onnx',
		'onnx_model_path': 'yolov8n.onnx',
		'onnx_int8_path': 'yolov8n.int8.onnx',
		'world_model': 'yolov8s-worldv2.pt',
		'hybrid_world_every': 8,
		'persistence_window': 5,
		'speech_cooldown': 2.5,
		'speech_gap': 1.5,
		'use_async_capture': True,
		'enable_distance_estimation': True,
		'enable_fall_detection': False,
		'pose_every': 10,
		'pose_model': 'yolov8n-pose.pt',
		'enable_speech': True,
		'prefer_int8': False,
	},
}


def _deep_merge(base: dict, override: dict):
	merged = deepcopy(base)
	for key, value in override.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _deep_merge(merged[key], value)
		else:
			merged[key] = value
	return merged


def load_runtime_settings(config_path: str):
	settings = deepcopy(FALLBACK_SETTINGS)
	path = Path(config_path)
	if not path.exists():
		return settings

	try:
		user_data = json.loads(path.read_text(encoding='utf-8'))
		if isinstance(user_data, dict):
			settings = _deep_merge(settings, user_data)
	except Exception as exc:
		print(f'Warning: failed to parse config file {path}: {exc}')

	return settings
