import unittest
import math
import struct

from locator_demo.audio import AudioDirection
from locator_demo.audio_device import FfmpegAudioLocalizer


class BlockingStdout:
    def read(self, _size):
        raise AssertionError("read_estimate must not block on ffmpeg stdout")


class FakeProcess:
    stdout = BlockingStdout()
    stderr = None

    def poll(self):
        return None


class AudioDeviceTests(unittest.TestCase):
    def test_read_estimate_uses_cached_background_result(self):
        localizer = FfmpegAudioLocalizer()
        localizer.start = lambda: None
        localizer._process = FakeProcess()

        estimate = localizer.read_estimate()

        self.assertEqual(estimate.direction, AudioDirection.UNKNOWN)

    def test_channel_level_bias_detects_close_side_speech(self):
        localizer = FfmpegAudioLocalizer()
        samples = []
        for i in range(160):
            wave = math.sin(i / 160.0 * math.tau * 4)
            samples.extend([int(wave * 26000), int(wave * 3000)])
        raw = struct.pack("<" + "h" * len(samples), *samples)

        estimate = localizer._estimate_from_raw(raw)

        self.assertEqual(estimate.direction, AudioDirection.LEFT)

    def test_low_voiced_speech_keeps_vad_confidence(self):
        localizer = FfmpegAudioLocalizer()
        sample_rate = localizer.sample_rate
        left = []
        right = []
        for i in range(int(sample_rate * 0.04)):
            value = 0.8 * math.sin(2 * math.pi * 160 * i / sample_rate)
            value += 0.35 * math.sin(2 * math.pi * 600 * i / sample_rate)
            value += 0.20 * math.sin(2 * math.pi * 1800 * i / sample_rate)
            left.append(value * 0.2)
            right.append(value * 0.2)

        confidence, noise_state = localizer._speech_confidence(left, right, energy=0.8)

        self.assertGreater(confidence, 0.35)
        self.assertEqual(noise_state, "voiced_speech")

    def test_srp_path_reports_front_azimuth(self):
        localizer = FfmpegAudioLocalizer()
        pulse = [0] * 64 + [24000] + [0] * 64
        left = [0] * 4 + pulse
        right = pulse + [0] * 4
        samples = []
        for l_value, r_value in zip(left, right):
            samples.extend([l_value, r_value])
        raw = struct.pack("<" + "h" * len(samples), *samples)

        estimate = localizer._estimate_from_raw(raw)

        self.assertEqual(estimate.direction, AudioDirection.LEFT)
        self.assertGreater(estimate.azimuth_deg, 0.0)
        self.assertGreater(estimate.doa_confidence, 0.12)


if __name__ == "__main__":
    unittest.main()
