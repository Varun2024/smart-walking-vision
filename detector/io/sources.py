import cv2
import numpy as np
import threading
import time
import urllib.request


def open_video_source(source: str):
	source = source.strip()

	if source.isdigit():
		cam_index = int(source)
		cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
		if not cap.isOpened():
			cap = cv2.VideoCapture(cam_index)
		return cap, False

	is_snapshot_url = source.lower().startswith('http') and source.lower().endswith(('.jpg', '.jpeg', '.png'))
	if is_snapshot_url:
		return None, True

	cap = cv2.VideoCapture(source)
	return cap, False


def configure_capture(cap, frame_width: int, frame_height: int):
	if cap is None:
		return
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
	cap.set(cv2.CAP_PROP_FPS, 60)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)


def read_frame(cap, source: str, snapshot_mode: bool):
	if snapshot_mode:
		img_resp = urllib.request.urlopen(source, timeout=5)
		img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
		frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
		if frame is None:
			return False, None
		return True, frame

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
