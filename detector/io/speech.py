import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from random import choice

import pygame
from gtts import gTTS


class SpeechAnnouncer:
	def __init__(self, cooldown_seconds: float = 2.5, min_gap_seconds: float = 1.5):
		self.cooldown_seconds = cooldown_seconds
		self.min_gap_seconds = min_gap_seconds
		self._queue: Queue[str] = Queue(maxsize=1)
		self._last_spoken: dict[str, float] = {}
		self._next_allowed_time = 0.0
		self._stop_event = threading.Event()
		self._worker = threading.Thread(target=self._run_worker, daemon=True)
		self._worker.start()

	def announce(self, label: str, count: int = 1):
		now = time.time()

		if now < self._next_allowed_time:
			return

		event_key = f'{label}:{count}'
		last_time = self._last_spoken.get(event_key, 0.0)
		if now - last_time < self.cooldown_seconds:
			return

		self._last_spoken[event_key] = now
		self._next_allowed_time = now + self.min_gap_seconds

		if self._queue.full():
			return

		self._queue.put_nowait(self._build_message(label, count))

	def close(self):
		self._stop_event.set()
		self._worker.join(timeout=1.5)
		if pygame.mixer.get_init():
			pygame.mixer.quit()

	def _run_worker(self):
		while not self._stop_event.is_set():
			try:
				text = self._queue.get(timeout=0.2)
			except Empty:
				continue

			try:
				self._speak_text(text)
			except Exception as exc:
				print(f'Speech error: {exc}')
			finally:
				self._queue.task_done()

	@staticmethod
	def _pluralize(noun: str) -> str:
		irregular = {
			'person': 'people',
			'man': 'men',
			'woman': 'women',
			'child': 'children',
		}
		if noun in irregular:
			return irregular[noun]
		if noun.endswith('s'):
			return noun
		return f'{noun}s'

	@staticmethod
	def _build_message(label: str, count: int) -> str:
		natural_label = label.replace('_', ' ').lower()

		if count > 1:
			plural_label = SpeechAnnouncer._pluralize(natural_label)
			templates = [
				f'There are now {count} {plural_label} in view.',
				f'I can see {count} {plural_label} ahead.',
				f'Count update: {count} {plural_label} detected.',
			]
			return choice(templates)

		templates = [
			f'I can see a {natural_label} ahead.',
			f'Heads up, detected {natural_label} in front of you.',
			f'{natural_label} detected nearby.',
			f'Noticed a {natural_label} ahead.',
		]
		return choice(templates)

	@staticmethod
	def _speak_text(text: str):
		tts = gTTS(text=text, lang='en', slow=False)

		with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
			temp_path = Path(temp_file.name)

		try:
			tts.save(str(temp_path))

			if not pygame.mixer.get_init():
				pygame.mixer.init()

			pygame.mixer.music.load(str(temp_path))
			pygame.mixer.music.play()

			while pygame.mixer.music.get_busy():
				time.sleep(0.05)

			pygame.mixer.music.unload()
		finally:
			if temp_path.exists():
				temp_path.unlink(missing_ok=True)
