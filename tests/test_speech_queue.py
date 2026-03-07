import time

from detector.io.speech import SpeechAnnouncer


def test_speech_queue_respects_global_gap(monkeypatch):
    monkeypatch.setattr(SpeechAnnouncer, '_speak_text', staticmethod(lambda text: time.sleep(0.01)))

    announcer = SpeechAnnouncer(cooldown_seconds=0.0, min_gap_seconds=5.0)
    try:
        announcer.announce('person', 1)
        assert 'person:1' in announcer._last_spoken

        announcer.announce('car', 1)
        assert 'car:1' not in announcer._last_spoken
    finally:
        announcer.close()


def test_speech_queue_respects_event_cooldown(monkeypatch):
    monkeypatch.setattr(SpeechAnnouncer, '_speak_text', staticmethod(lambda text: time.sleep(0.01)))

    announcer = SpeechAnnouncer(cooldown_seconds=10.0, min_gap_seconds=0.0)
    try:
        announcer.announce('person', 1)
        first_ts = announcer._last_spoken.get('person:1')
        assert first_ts is not None

        time.sleep(0.01)
        announcer.announce('person', 1)
        assert announcer._last_spoken['person:1'] == first_ts
    finally:
        announcer.close()
