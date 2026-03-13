import cv2
import numpy as np
import threading
import time
import urllib.request


def _resolve_http_source(source: str):
	lower_source = source.lower()
	if not lower_source.startswith(('http://', 'https://')):
		return source, False

	if lower_source.endswith(('.jpg', '.jpeg', '.png')):
		return source, True

	# Support ESP32-CAM base URLs like http://<ip>/ by defaulting to a high-res snapshot endpoint.
	if source.endswith('/'):
		return f'{source}cam-hi.jpg', True

	if source.rsplit('/', 1)[-1].find('.') == -1:
		return f'{source}/cam-hi.jpg', True

	return source, False


def open_video_source(source: str):
	source = source.strip()

	if source.isdigit():
		cam_index = int(source)
		cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
		if not cap.isOpened():
			cap = cv2.VideoCapture(cam_index)
		return cap, False, source

	source, is_snapshot_url = _resolve_http_source(source)
	if is_snapshot_url:
		return None, True, source

	cap = cv2.VideoCapture(source)
	return cap, False, source


def configure_capture(cap, frame_width: int, frame_height: int):
	if cap is None:
		return
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
	cap.set(cv2.CAP_PROP_FPS, 60)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)


def read_frame(cap, source: str, snapshot_mode: bool, snapshot_timeout_seconds: float = 5.0):
	if snapshot_mode:
		try:
			img_resp = urllib.request.urlopen(source, timeout=max(0.1, float(snapshot_timeout_seconds)))
			img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
			frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
			if frame is None:
				return False, None
			return True, frame
		except Exception:
			# Snapshot endpoints can time out intermittently on weak Wi-Fi links.
			return False, None

	return cap.read()


class AsyncVideoCapture:
	def __init__(self, cap):
		self.cap = cap
		self._lock = threading.Lock()
		self._latest_frame = None
		self._latest_success = False
		self._stopped = threading.Event()
		self._thread = threading.Thread(target=self._reader_loop, daemon=True)

	def start(self):
		self._thread.start()
		return self

	def read(self):
		with self._lock:
			if self._latest_frame is None:
				return False, None
			return self._latest_success, self._latest_frame.copy()

	def stop(self):
		self._stopped.set()
		self._thread.join(timeout=1.0)

	def _reader_loop(self):
		while not self._stopped.is_set():
			success, frame = self.cap.read()
			if success and frame is not None:
				with self._lock:
					self._latest_success = True
					self._latest_frame = frame
			else:
				with self._lock:
					self._latest_success = False
				time.sleep(0.005)
