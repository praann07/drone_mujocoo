"""Stage E voice input sources (docs/ENGINEERING_PLAN.md Stage E task 1).

Defines a common interface: ``get_command() -> (command, meta)`` where
``command`` is one of the 6 fixed commands or None (no utterance yet), and
``meta`` carries the raw transcript, ASR confidence, and latency breakdown
in milliseconds (vad/asr/classify/trajectory/dispatch/total).

Two implementations:
  * ``VoskSource``  - real microphone capture via sounddevice + Vosk ASR,
                      with a simple energy/silence-based utterance ender.
  * ``TextSource``  - feeds pre-scripted command strings (or stdin lines),
                      used for headless trials and dev work without a mic.

Both push completed utterances onto a queue so the render/control loop can
drain them without blocking.
"""
from __future__ import annotations

import queue
import threading
import time

from commands import classify, COMMANDS


class _BaseSource:
    def __init__(self):
        self._queue: "queue.Queue[tuple[str | None, dict]]" = queue.Queue()

    def get_command(self, timeout: float = 0.0):
        """Return (command, meta) for the next completed utterance, or
        (None, {}) if none is ready within ``timeout`` seconds."""
        try:
            if timeout <= 0.0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None, {}

    def _emit(self, transcript: str, confidence: float, latencies_ms: dict):
        """Classify `transcript` and push (command, meta) onto the queue.

        `classify` is timed HERE, once, centrally - not by each caller -
        so every source (Vosk, text) reports a real measured classify cost
        instead of a caller-supplied guess, and classify() is never run
        twice for the same utterance (once to "measure" it, once for
        real) as a prior version of this file did."""
        t0 = time.perf_counter()
        cmd = classify(transcript)
        latencies_ms = dict(latencies_ms)
        latencies_ms["classify"] = 1000.0 * (time.perf_counter() - t0)
        total = sum(latencies_ms.get(k, 0.0) for k in
                    ("vad", "asr", "classify", "trajectory", "dispatch"))
        latencies_ms["total"] = total
        meta = {
            "transcript": transcript,
            "asr_confidence": confidence,
            "latencies_ms": latencies_ms,
        }
        self._queue.put((cmd, meta))


class TextSource(_BaseSource):
    """Feeds commands from an iterable of strings (one per utterance), or,
    if ``interactive=True``, from stdin lines. classify() runs on each line
    so transcripts need not be exact commands (e.g. "go forward please")."""

    def __init__(self, scripted=None, interactive: bool = False):
        super().__init__()
        self._scripted = list(scripted) if scripted else []
        self._interactive = interactive
        self._idx = 0
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self):
        if self._scripted:
            for line in self._scripted:
                # vad=0, asr=0: there is no audio pipeline in this source at
                # all (it feeds text directly), not a stubbed measurement of
                # a real stage - see the README caveat on scripted-vs-live
                # latency numbers. classify cost is measured for real by
                # _emit(), not simulated here.
                self._emit(line, 1.0, {"vad": 0.0, "asr": 0.0})
                time.sleep(0.05)
            return
        if self._interactive:
            import sys
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    return
                if not line:
                    return
                self._emit(line, 1.0, {"vad": 0.0, "asr": 0.0})


class VoskSource(_BaseSource):
    """Microphone capture + Vosk ASR. Requires a Vosk model directory
    (default: env VOSK_MODEL_DIR, else ./vosk-model-en-us). Runs a
    background thread reading 16 kHz mono audio; an utterance ends after a
    short silence gap, at which point the final transcript is classified."""

    def __init__(self, model_dir: str | None = None, sample_rate: int = 16000,
                 silence_s: float = 0.4, device: int | None = None):
        super().__init__()
        import os
        model_dir = model_dir or os.environ.get("VOSK_MODEL_DIR")
        if not model_dir:
            raise RuntimeError(
                "VoskSource needs a model dir: set VOSK_MODEL_DIR or pass model_dir= "
                "(download vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models)")
        from vosk import Model, KaldiRecognizer
        import sounddevice as sd

        self._model = Model(model_dir)
        self._sample_rate = sample_rate
        self._silence_s = silence_s
        self._rec = KaldiRecognizer(self._model, sample_rate)

        # Resolve the device index HERE, in this process, right before
        # opening the stream - not from a number the caller looked up in a
        # separate earlier process. On a machine with multiple Bluetooth
        # audio endpoints churning, PortAudio's device index<->hardware
        # mapping is only valid within the process that queried it; an
        # index fetched a moment ago in a different `python -c "..."`
        # invocation is stale by the time this process starts and raises
        # 'Invalid device' - matching exactly what happened here. `device`
        # (if given) is used as a hint only when a device with that exact
        # index still exists in THIS process's own live enumeration.
        device = self._resolve_device(device)
        self._stream = sd.RawInputStream(
            samplerate=sample_rate, blocksize=4096, dtype="int16",
            channels=1, callback=self._audio_callback, device=device)
        self._lock = threading.Lock()
        self._partial = ""
        self._last_speech = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    @staticmethod
    def _resolve_device(requested: int | None) -> int | None:
        """Pick an input device index using a LIVE enumeration in this
        process, never a number carried over from elsewhere. Preference
        order: (1) `requested`, but only if it's still a valid input
        device right now; (2) the first input-capable device whose name
        does NOT look like a Bluetooth Hands-Free endpoint
        (`bthhfenum.sys`), since those are the ones observed flapping in
        and out of the device list here; (3) `None` (PortAudio's own
        default), as a last resort."""
        import sounddevice as sd
        devices = sd.query_devices()

        if requested is not None:
            if 0 <= requested < len(devices) and devices[requested]["max_input_channels"] > 0:
                return requested
            print(f"[VoskSource] Requested device {requested} is not a valid input "
                  f"device in this process's current enumeration (device list has "
                  f"changed since it was looked up) - falling back to auto-selection.",
                  flush=True)

        non_bt = [i for i, d in enumerate(devices)
                  if d["max_input_channels"] > 0 and "bthhfenum" not in d["name"].lower()]
        if non_bt:
            print(f"[VoskSource] Auto-selected non-Bluetooth input device "
                  f"{non_bt[0]}: {devices[non_bt[0]]['name']!r}", flush=True)
            return non_bt[0]

        any_input = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if any_input:
            print(f"[VoskSource] No non-Bluetooth input device found; using "
                  f"{any_input[0]}: {devices[any_input[0]]['name']!r}", flush=True)
            return any_input[0]

        return None  # let PortAudio pick its own default and fail loudly if that's also bad

    def _audio_callback(self, indata, frames, time_info, status):
        import time as _t
        data = bytes(indata)
        with self._lock:
            t0 = _t.perf_counter()
            accepted = self._rec.AcceptWaveform(data)
            if accepted:
                res = self._rec.Result()
                # This callback IS the ASR stage: AcceptWaveform() decodes
                # this audio chunk against the acoustic+language model, and
                # Result() finalizes the utterance. There is no separate
                # VAD step to time - Kaldi's recognizer performs endpoint
                # detection internally as part of decoding, so "vad" stays
                # 0 by construction (no distinct stage exists), not because
                # it was measured and happened to be free.
                asr_ms = 1000.0 * (_t.perf_counter() - t0)
                self._maybe_emit(res, asr_ms)
            else:
                self._partial = self._rec.PartialResult()
        self._last_speech = _t.perf_counter()

    def _maybe_emit(self, result_json: str, asr_ms: float):
        import json
        try:
            obj = json.loads(result_json)
        except Exception:
            return
        text = obj.get("text", "").strip()
        conf = float(obj.get("result", [{}])[0].get("conf", 1.0)) if obj.get("result") else 1.0
        if not text:
            return
        self._emit(text, conf, {"vad": 0.0, "asr": asr_ms})

    def _pump(self):
        import time as _t
        while self._running:
            _t.sleep(0.05)

    def start(self, retries: int = 3, retry_delay_s: float = 1.5):
        """Start capture, retrying past the specific failure this Bluetooth
        headset actually hit: PaErrorCode -9999 / WdmSyncIoctl. A Bluetooth
        headset commonly idles in output-only (A2DP) profile and only
        renegotiates to the mic-capable (HFP) profile once something
        requests capture - that negotiation takes a beat, and opening the
        stream before it completes is exactly this error, not a
        disconnected device (a truly disconnected device fails instantly
        and every retry, which still surfaces via the message below)."""
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                self._stream.start()
                return
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(retry_delay_s)
        raise RuntimeError(
            f"\n[Microphone Error] Could not start audio input stream after "
            f"{retries} attempts (with {retry_delay_s}s between tries to let "
            f"a Bluetooth headset finish switching to its mic-capable "
            f"profile): {last_err}\n"
            "Fix:\n"
            "  1. Make a sound near / speak to the headset once to nudge Windows into "
            "switching profiles, then rerun, OR\n"
            "  2. Reconnect/turn on your Bluetooth headset or plug in a wired microphone, OR\n"
            "  3. Select an active microphone under Windows 'Settings > System > Sound > Input', OR\n"
            "  4. Use '--source text' to control the drone immediately via keyboard typing."
        ) from None

    def stop(self):
        self._running = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
