from __future__ import annotations

import math
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .audio import AudioDirectionSmoother, AudioEstimate, classify_audio_direction, rms, srp_phat_front_hemisphere
from .devices import choose_dshow_device


@dataclass(frozen=True)
class AudioDeviceStatus:
    ok: bool
    device_name: str
    message: str


class SimulatedAudioLocalizer:
    def __init__(self):
        self._phase = 0

    def read_estimate(self) -> AudioEstimate:
        self._phase = (self._phase + 1) % 120
        tdoa = math.sin(self._phase / 120.0 * math.tau) * 0.00022
        return classify_audio_direction(tdoa, 0.75)

    def status(self) -> AudioDeviceStatus:
        return AudioDeviceStatus(True, "simulated", "simulated audio localizer")


class FfmpegAudioLocalizer:
    def __init__(
        self,
        *,
        device_name: str | None = None,
        sample_rate: int = 16000,
        frame_ms: int = 40,
        ffmpeg: str = "ffmpeg",
    ):
        self.requested_device_name = device_name
        self.device_name = device_name or ""
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.ffmpeg = ffmpeg
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._latest_error = "stopped"
        self._reader: threading.Thread | None = None
        self._smoother = AudioDirectionSmoother(window_size=4, min_samples=2)
        self._latest_estimate = classify_audio_direction(0.0, 0.0)
        self._latest_frame_time = 0.0
        self._noise_floor = 0.03
        history_frames = max(1, int(math.ceil(3.5 * 1000.0 / max(1, self.frame_ms))))
        self._recent_raw: deque[tuple[float, bytes]] = deque(maxlen=history_frames)

    def _resolve_device_name(self) -> str | None:
        if self.device_name:
            return self.device_name
        device = choose_dshow_device(
            "audio",
            [self.requested_device_name, "YDM2MIC", "Realtek", "Virtual Desktop Audio"],
            ffmpeg=self.ffmpeg,
        )
        if device is None:
            self._latest_error = "no DirectShow audio input found"
            return None
        self.device_name = device.name
        return self.device_name

    def _consume_stderr_if_exited(self) -> None:
        if self._process is None or self._process.poll() is None or self._process.stderr is None:
            return
        stderr = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        if stderr:
            self._latest_error = stderr.splitlines()[-1]
        else:
            self._latest_error = f"ffmpeg exited with {self._process.returncode}"

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        device_name = self._resolve_device_name()
        if device_name is None:
            return
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dshow",
            "-i",
            f"audio={device_name}",
            "-ac",
            "2",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "-",
        ]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._latest_error = "starting"
        except Exception as exc:
            self._process = None
            self._latest_error = f"audio error: {exc}"
            return
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _speech_confidence(self, left: list[float], right: list[float], energy: float) -> tuple[float, str]:
        try:
            import numpy as np

            mixed = (np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)) * 0.5
            if mixed.size < 8:
                return 0.0, "quiet"
            spectrum = np.abs(np.fft.rfft(mixed))
            freqs = np.fft.rfftfreq(mixed.size, 1.0 / self.sample_rate)
            full = float(np.sum(spectrum * spectrum)) + 1e-12
            voiced_band = (freqs >= 80.0) & (freqs < 250.0)
            speech_band = (freqs >= 250.0) & (freqs <= 1000.0)
            clarity_band = (freqs > 1000.0) & (freqs <= 4000.0)
            rumble_band = (freqs >= 20.0) & (freqs < 80.0)
            low_motor_band = (freqs >= 80.0) & (freqs < 180.0)
            voiced_ratio = float(np.sum(spectrum[voiced_band] * spectrum[voiced_band]) / full)
            speech_ratio = float(np.sum(spectrum[speech_band] * spectrum[speech_band]) / full)
            clarity_ratio = float(np.sum(spectrum[clarity_band] * spectrum[clarity_band]) / full)
            rumble_ratio = float(np.sum(spectrum[rumble_band] * spectrum[rumble_band]) / full)
            low_motor_ratio = float(np.sum(spectrum[low_motor_band] * spectrum[low_motor_band]) / full)
            noise_margin = energy / max(self._noise_floor * 1.6, 1e-4)
            speech_score = 0.90 * voiced_ratio + 1.45 * speech_ratio + 1.20 * clarity_ratio
            confidence = min(1.0, max(0.0, speech_score)) * min(1.0, noise_margin)
            if rumble_ratio > 0.55 and speech_ratio + clarity_ratio < 0.28:
                confidence *= 0.35
                return confidence, "fan_or_motor"
            if low_motor_ratio > 0.72 and clarity_ratio < 0.05 and speech_ratio < 0.12:
                confidence *= 0.55
                return confidence, "low_band_noise"
            if energy < max(0.045, self._noise_floor * 1.4):
                return min(confidence, 0.18), "quiet"
            if voiced_ratio >= 0.20 and speech_ratio + clarity_ratio >= 0.12:
                confidence = max(confidence, min(1.0, 0.25 + voiced_ratio + 0.5 * (speech_ratio + clarity_ratio)))
                return confidence, "voiced_speech"
            return confidence, "speech_like" if confidence >= 0.35 else "noise"
        except Exception:
            margin = energy / max(self._noise_floor * 1.6, 1e-4)
            confidence = min(1.0, max(0.0, (margin - 0.6) / 1.8))
            return confidence, "speech_like" if confidence >= 0.35 else "noise"

    def _estimate_from_raw(self, raw: bytes) -> AudioEstimate:
        values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
        left = [values[i] / 32768.0 for i in range(0, len(values), 2)]
        right = [values[i] / 32768.0 for i in range(1, len(values), 2)]
        left_level = rms(left)
        right_level = rms(right)
        energy = max(left_level, right_level) * 12.0
        speech_confidence, noise_state = self._speech_confidence(left, right, energy)
        if speech_confidence < 0.20:
            self._noise_floor = 0.98 * self._noise_floor + 0.02 * max(energy, 0.001)
        level_sum = left_level + right_level
        if energy >= 0.08 and level_sum > 0:
            balance = (left_level - right_level) / level_sum
            if speech_confidence >= 0.25 and abs(balance) >= 0.30:
                tdoa = 0.00035 if balance > 0 else -0.00035
                return classify_audio_direction(
                    tdoa,
                    min(1.0, energy),
                    min_energy=0.08,
                    speech_confidence=speech_confidence,
                    doa_confidence=0.65,
                    peak_ratio=1.0 + abs(balance) * 3.0,
                    noise_state=noise_state,
                )
        srp = srp_phat_front_hemisphere(left, right, sample_rate=self.sample_rate)
        return classify_audio_direction(
            srp.tdoa_s,
            min(1.0, energy),
            min_energy=0.08,
            speech_confidence=speech_confidence,
            doa_confidence=srp.confidence,
            peak_ratio=srp.peak_ratio,
            noise_state=noise_state,
            azimuth_deg=srp.azimuth_deg,
        )

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        sample_count = max(1, int(self.sample_rate * self.frame_ms / 1000))
        frame_bytes = sample_count * 2 * 2
        while process.poll() is None:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                self._consume_stderr_if_exited()
                time.sleep(0.02)
                continue
            estimate = self._smoother.update(self._estimate_from_raw(raw))
            with self._lock:
                self._recent_raw.append((time.time(), raw))
                self._latest_estimate = estimate
                self._latest_frame_time = time.time()
                self._latest_error = "running"

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def status(self) -> AudioDeviceStatus:
        running = self._process is not None and self._process.poll() is None
        if not running:
            self._consume_stderr_if_exited()
        return AudioDeviceStatus(running, self.device_name or "auto", "running" if running else self._latest_error)

    def read_estimate(self) -> AudioEstimate:
        with self._lock:
            self.start()
            if self._process is None:
                return classify_audio_direction(0.0, 0.0)
            return self._latest_estimate

    def recent_mono_audio(self, *, seconds: float = 2.0) -> tuple[int, list[float]]:
        cutoff = time.time() - max(0.1, float(seconds))
        with self._lock:
            chunks = [raw for ts, raw in self._recent_raw if ts >= cutoff]
        if not chunks:
            return self.sample_rate, []
        values = struct.unpack("<" + "h" * (sum(len(chunk) for chunk in chunks) // 2), b"".join(chunks))
        mono = [
            ((values[i] + values[i + 1]) * 0.5) / 32768.0
            for i in range(0, len(values) - 1, 2)
        ]
        return self.sample_rate, mono


AudioFactory = Callable[[], FfmpegAudioLocalizer | SimulatedAudioLocalizer]
