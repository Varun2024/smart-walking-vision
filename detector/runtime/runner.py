import argparse
import time
from pathlib import Path
import zlib

import cv2

from .config import DEFAULT_SOURCE, DEFAULT_SNAPSHOT_URL, DetectorConfig
from .runtime_settings import load_runtime_settings
from ..backends.merge_utils import merge_detections
from ..backends.yolo import YoloDetector
from ..backends.yolo_onnx import YoloOnnxDetector
from ..backends.yolo_world import YoloWorldDetector
from ..io.sources import AsyncVideoCapture, configure_capture, open_video_source, read_frame
from ..io.speech import SpeechAnnouncer
from ..safety.pose import FallDetector
from ..safety.temporal import CountPersistence, TemporalSmoother, estimate_proximity_label


def parse_args():
	parser = argparse.ArgumentParser(description='Object detection test (YOLOv3, YOLOv8 ONNX, YOLO-World, or hybrid)')
	parser.add_argument('--config', default='config/runtime_defaults.json', help='Path to runtime JSON config file')
	parser.add_argument(
		'--source',
		default=DEFAULT_SOURCE,
		help='Video source: webcam index (0), stream/file path, or snapshot URL (http://.../cam-hi.jpg)',
	)
	parser.add_argument('--model-type', choices=['yolov3', 'yolov8onnx', 'yoloworld', 'hybrid'], default=None, help='Detection backend to use')
	parser.add_argument('--onnx-model-path', default=None, help='Path to YOLOv8 ONNX model file')
	parser.add_argument('--onnx-int8-path', default=None, help='Path to INT8 quantized ONNX model file')
	parser.add_argument('--prefer-int8', action='store_true', help='Use INT8 ONNX model when available to reduce latency')
	parser.add_argument('--world-model', default=None, help='YOLO-World model name/path for open-vocabulary detection')
	parser.add_argument('--world-classes', default=None, help='Comma-separated classes for YOLO-World (example: person,watch,cell phone)')
	parser.add_argument('--quality', choices=['fast', 'balanced', 'accurate'], default=None, help='Preset that balances speed vs detection accuracy')
	parser.add_argument('--model-size', type=int, default=None, help='YOLO input size (e.g. 320, 416, 608)')
	parser.add_argument('--confidence', type=float, default=None, help='Minimum confidence score to keep detections')
	parser.add_argument('--nms-threshold', type=float, default=None, help='NMS overlap threshold for suppressing boxes')
	parser.add_argument('--width', type=int, default=None, help='Capture width for camera/video source')
	parser.add_argument('--height', type=int, default=None, help='Capture height for camera/video source')
	parser.add_argument('--infer-every', type=int, default=None, help='Run model every N frames (higher is faster but less responsive)')
	parser.add_argument('--hybrid-world-every', type=int, default=None, help='In hybrid mode, run YOLO-World once every N inference cycles')
	parser.add_argument('--persistence-window', type=int, default=None, help='Frames used to smooth object count flicker')
	parser.add_argument('--no-async-capture', action='store_true', help='Disable threaded frame capture')
	parser.add_argument('--disable-distance-estimation', action='store_true', help='Disable proximity estimation overlay')
	parser.add_argument('--enable-fall-detection', action='store_true', help='Enable pose-based fall-risk detection')
	parser.add_argument('--pose-every', type=int, default=None, help='Run pose model once every N inference cycles')
	parser.add_argument('--pose-model', default=None, help='Pose model for fall detection')
	parser.add_argument('--speech-cooldown', type=float, default=None, help='Seconds before repeating same label')
	parser.add_argument('--speech-gap', type=float, default=None, help='Minimum seconds between any two spoken messages')
	parser.add_argument('--no-speech', action='store_true', help='Disable gTTS announcements')
	return parser.parse_args()


def build_config(args) -> DetectorConfig:
	settings = load_runtime_settings(args.config)
	defaults = settings['defaults']
	quality_presets = settings['quality_presets']
	world_default_classes = tuple(settings['world_default_classes'])

	source = DEFAULT_SNAPSHOT_URL if args.source.lower() == 'espcam' else args.source

	quality = args.quality if args.quality is not None else defaults['quality']
	preset = quality_presets[quality]

	model_type = args.model_type if args.model_type is not None else defaults['model_type']
	onnx_model_path = args.onnx_model_path if args.onnx_model_path is not None else defaults['onnx_model_path']
	onnx_int8_path = args.onnx_int8_path if args.onnx_int8_path is not None else defaults['onnx_int8_path']
	world_model = args.world_model if args.world_model is not None else defaults['world_model']

	model_size = args.model_size if args.model_size is not None else preset['model_size']
	confidence = args.confidence if args.confidence is not None else preset['confidence']
	nms_threshold = args.nms_threshold if args.nms_threshold is not None else preset['nms_threshold']
	width = args.width if args.width is not None else preset['width']
	height = args.height if args.height is not None else preset['height']
	infer_every = args.infer_every if args.infer_every is not None else preset['infer_every']
	infer_every = max(1, infer_every)
	hybrid_world_every_default = int(defaults['hybrid_world_every'])
	hybrid_world_every = max(1, args.hybrid_world_every if args.hybrid_world_every is not None else hybrid_world_every_default)
	persistence_window_default = int(defaults['persistence_window'])
	persistence_window = max(1, args.persistence_window if args.persistence_window is not None else persistence_window_default)
	speech_cooldown = args.speech_cooldown if args.speech_cooldown is not None else float(defaults['speech_cooldown'])
	speech_gap = args.speech_gap if args.speech_gap is not None else float(defaults['speech_gap'])
	pose_every_default = int(defaults['pose_every'])
	pose_every = max(1, args.pose_every if args.pose_every is not None else pose_every_default)
	pose_model = args.pose_model if args.pose_model is not None else defaults['pose_model']

	use_async_capture = bool(defaults['use_async_capture'])
	if args.no_async_capture:
		use_async_capture = False

	enable_distance_estimation = bool(defaults['enable_distance_estimation'])
	if args.disable_distance_estimation:
		enable_distance_estimation = False

	enable_fall_detection = bool(defaults['enable_fall_detection']) or args.enable_fall_detection
	prefer_int8 = bool(defaults['prefer_int8']) or args.prefer_int8
	enable_speech = bool(defaults['enable_speech'])
	if args.no_speech:
		enable_speech = False

	if args.world_classes:
		world_classes = tuple(part.strip() for part in args.world_classes.split(',') if part.strip())
	else:
		world_classes = world_default_classes

	return DetectorConfig(
		source=source,
		model_cfg=Path('yolov3.cfg'),
		model_weights=Path('yolov3.weights'),
		class_names=Path('coco.names'),
		model_type=model_type,
		onnx_model=Path(onnx_model_path),
		onnx_int8_model=Path(onnx_int8_path),
		prefer_int8=prefer_int8,
		world_model=world_model,
		world_classes=world_classes,
		confidence=confidence,
		nms_threshold=nms_threshold,
		model_size=model_size,
		frame_width=width,
		frame_height=height,
		infer_every_n_frames=infer_every,
		hybrid_world_every_n_frames=hybrid_world_every,
		persistence_window=persistence_window,
		use_async_capture=use_async_capture,
		enable_distance_estimation=enable_distance_estimation,
		enable_fall_detection=enable_fall_detection,
		pose_every_n_frames=pose_every,
		pose_model=pose_model,
		speech_cooldown_seconds=speech_cooldown,
		speech_gap_seconds=speech_gap,
		enable_speech=enable_speech,
	)


def _color_for_class(class_name: str):
	seed = zlib.crc32(class_name.encode('utf-8'))
	blue = 80 + (seed & 0x7F)
	green = 80 + ((seed >> 8) & 0x7F)
	red = 80 + ((seed >> 16) & 0x7F)
	return int(blue), int(green), int(red)


def _draw_proximity(frame, detections):
	max_urgency = -1
	urgent_label = None
	for class_name, _, (x, y, width, height) in detections:
		proximity_label, urgency = estimate_proximity_label(frame.shape, (x, y, width, height))
		color = _color_for_class(class_name)
		cv2.putText(
			frame,
			proximity_label,
			(x, y + height + 18),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			color,
			2,
		)
		if urgency > max_urgency:
			max_urgency = urgency
			urgent_label = f'{class_name} {proximity_label}'

	if urgent_label is not None and max_urgency >= 2:
		cv2.putText(
			frame,
			f'URGENT: {urgent_label}',
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 0, 255),
			2,
		)


def _initialize_primary_detector(config: DetectorConfig):
	if config.model_type == 'yolov3':
		return YoloDetector(
			cfg_path=config.model_cfg,
			weights_path=config.model_weights,
			class_names_path=config.class_names,
			confidence=config.confidence,
			nms_threshold=config.nms_threshold,
			model_size=config.model_size,
		)

	if config.model_type in ('yolov8onnx', 'hybrid'):
		selected_onnx = config.onnx_model
		if config.prefer_int8 and config.onnx_int8_model.exists():
			selected_onnx = config.onnx_int8_model

		return YoloOnnxDetector(
			onnx_model_path=selected_onnx,
			class_names_path=config.class_names,
			confidence=config.confidence,
			nms_threshold=config.nms_threshold,
			model_size=config.model_size,
		)

	if config.model_type == 'yoloworld':
		return YoloWorldDetector(
			world_model_name=config.world_model,
			class_prompts=config.world_classes,
			confidence=config.confidence,
			nms_threshold=config.nms_threshold,
			model_size=config.model_size,
		)

	raise ValueError(f'Unsupported model type: {config.model_type}')


def _initialize_world_detector(config: DetectorConfig):
	if config.model_type != 'hybrid':
		return None

	return YoloWorldDetector(
		world_model_name=config.world_model,
		class_prompts=config.world_classes,
		confidence=config.confidence,
		nms_threshold=config.nms_threshold,
		model_size=config.model_size,
	)


def run():
	args = parse_args()
	config = build_config(args)
	detector = None
	world_detector = None
	pose_detector = None

	try:
		detector = _initialize_primary_detector(config)
	except FileNotFoundError as exc:
		print(exc)
		print('Run: python -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\').export(format=\'onnx\', imgsz=640, opset=12, simplify=True)"')
		return
	except Exception as exc:
		print(f'Failed to initialize detector: {exc}')
		return

	if config.enable_fall_detection:
		try:
			pose_detector = FallDetector(model_path=config.pose_model)
		except Exception as exc:
			print(f'Pose model initialization failed, continuing without fall detection: {exc}')
			pose_detector = None

	if config.model_type == 'hybrid':
		try:
			world_detector = _initialize_world_detector(config)
		except Exception as exc:
			print(f'Failed to initialize YOLO-World for hybrid mode: {exc}')
			print('Tip: ensure ultralytics package is installed and internet is available for first model download.')
			return

	cap, snapshot_mode = open_video_source(config.source)
	if not snapshot_mode:
		configure_capture(cap, config.frame_width, config.frame_height)
		if cap is None or not cap.isOpened():
			print(f'Unable to open source: {config.source}')
			return

	async_capture = None
	if not snapshot_mode and config.use_async_capture:
		async_capture = AsyncVideoCapture(cap).start()

	announcer = (
		SpeechAnnouncer(
			cooldown_seconds=config.speech_cooldown_seconds,
			min_gap_seconds=config.speech_gap_seconds,
		)
		if config.enable_speech
		else None
	)
	previous_counts = {}
	active_detections = []
	active_counts = {}
	smoothed_counts = {}
	frame_index = 0
	fps = 0.0
	last_time = time.time()
	infer_cycle = 0
	temporal_smoother = TemporalSmoother(iou_threshold=0.35)
	count_persistence = CountPersistence(history_size=config.persistence_window)

	try:
		while True:
			try:
				if async_capture is not None:
					success, frame = async_capture.read()
				else:
					success, frame = read_frame(cap, config.source, snapshot_mode)
			except Exception as exc:
				print(f'Frame read error: {exc}')
				break

			if not success:
				if async_capture is not None:
					time.sleep(0.005)
					continue
				print('No frame received from source.')
				break

			frame_index += 1
			should_infer = (frame_index - 1) % config.infer_every_n_frames == 0

			if should_infer:
				infer_cycle += 1

				outputs = detector.infer(frame)
				primary_detections = detector.postprocess(outputs, frame)

				if config.model_type == 'hybrid' and world_detector is not None:
					should_scan_world = infer_cycle % config.hybrid_world_every_n_frames == 0
					if should_scan_world:
						world_outputs = world_detector.infer(frame)
						world_detections = world_detector.postprocess(world_outputs, frame)
						active_detections = merge_detections(primary_detections, world_detections, iou_threshold=0.5)
					else:
						active_detections = primary_detections
				else:
					active_detections = primary_detections

				temporal_smoother.update(active_detections)
				active_counts = detector.count_by_class(active_detections)
				smoothed_counts = count_persistence.smooth(active_counts)

				if announcer is not None:
					for class_name, current_count in smoothed_counts.items():
						previous_count = previous_counts.get(class_name, 0)
						if current_count > previous_count:
							announcer.announce(class_name, current_count)

				if smoothed_counts != previous_counts:
					summary = ', '.join(f'{name}:{count}' for name, count in sorted(smoothed_counts.items()))
					print(f'Detected counts -> {summary}' if summary else 'Detected counts -> none')

				previous_counts = smoothed_counts
			else:
				active_detections = temporal_smoother.predict()
				active_counts = detector.count_by_class(active_detections)
				smoothed_counts = count_persistence.smooth(active_counts)

			fall_box = None
			if pose_detector is not None and should_infer and infer_cycle % config.pose_every_n_frames == 0:
				fall_detected, fall_box = pose_detector.detect_fall(frame)
				if fall_detected and announcer is not None and pose_detector.should_alert(cooldown_seconds=6.0):
					announcer.announce('person fallen', 1)

			if fall_box is not None:
				pose_detector.draw_alert(frame, fall_box)

			detector.draw_detections(frame, active_detections)
			detector.draw_count_summary(frame, smoothed_counts)

			if config.enable_distance_estimation:
				_draw_proximity(frame, active_detections)

			current_time = time.time()
			delta = max(current_time - last_time, 1e-6)
			instant_fps = 1.0 / delta
			fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
			last_time = current_time

			cv2.putText(
				frame,
				f'FPS: {fps:.1f} | Quality: {args.quality} | Mode: {config.model_type} | Infer every: {config.infer_every_n_frames} | Persist: {config.persistence_window}',
				(10, frame.shape[0] - 12),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(0, 255, 0),
				2,
			)

			cv2.imshow('Object identification', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		if async_capture is not None:
			async_capture.stop()
		if cap is not None:
			cap.release()
		cv2.destroyAllWindows()
		if announcer is not None:
			announcer.close()
