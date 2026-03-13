from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE = '0'
DEFAULT_SNAPSHOT_URL = 'http://192.168.254.63/cam-hi.jpg'


@dataclass(frozen=True)
class DetectorConfig:
	source: str
	model_cfg: Path
	model_weights: Path
	class_names: Path
	model_type: str = 'yolov8onnx'
	onnx_model: Path = Path('yolov8n.onnx')
	onnx_int8_model: Path = Path('yolov8n.int8.onnx')
	prefer_int8: bool = False
	world_model: str = 'yolov8s-worldv2.pt'
	world_classes: tuple[str, ...] = ()
	confidence: float = 0.45
	nms_threshold: float = 0.4
	model_size: int = 416
	frame_width: int = 1280
	frame_height: int = 720
	infer_every_n_frames: int = 1
	hybrid_world_every_n_frames: int = 8
	persistence_window: int = 5
	use_async_capture: bool = True
	enable_distance_estimation: bool = True
	enable_fall_detection: bool = False
	pose_every_n_frames: int = 10
	pose_model: str = 'yolov8n-pose.pt'
	speech_cooldown_seconds: float = 2.5
	speech_gap_seconds: float = 1.5
	enable_speech: bool = True
	snapshot_timeout_seconds: float = 2.0
	snapshot_retry_delay_seconds: float = 0.08
