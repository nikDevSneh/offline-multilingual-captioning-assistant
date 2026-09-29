# Offline Multilingual Captioning & Transcription Assistant

Real-time, on-device captioning for Hindi / English / code-switched speech,
built to run on the Hexagon NPU of Snapdragon-powered HP PCs via Qualcomm AI Hub.

Everything in this repo runs **fully offline** once the model is exported — no
audio or text ever leaves the machine.

---

## Project layout

```
captioning_assistant/
├── config.yaml                  # all runtime settings
├── main.py                      # entry point
├── requirements.txt             # cross-platform dependencies
├── requirements-snapdragon.txt  # extra deps only needed on the Snapdragon target
├── src/
│   ├── audio_capture.py         # mic capture + overlapping chunker
│   ├── asr_engine.py            # Whisper engines: QNN (NPU) + CPU dev fallback
│   ├── translator.py            # optional Hindi <-> English translation
│   ├── caption_overlay.py       # always-on-top caption window (Tkinter)
│   ├── transcript_writer.py     # session transcript -> .txt / .srt
│   └── app.py                   # wires everything together
├── scripts/
│   ├── export_whisper_qnn.py    # exports & quantizes Whisper via Qualcomm AI Hub
│   └── benchmark.py             # NPU vs CPU latency/power comparison
└── tests/
    ├── test_chunker.py
    └── test_transcript_writer.py
```

## Two ways to run this

**1. Development machine (any OS, no Snapdragon needed)**
Use the CPU fallback engine (`faster-whisper`) to build and test the app logic —
audio capture, chunking, overlay UI, transcript export — before touching real
Snapdragon hardware.

**2. Snapdragon-powered HP PC (the real target)**
Export the quantized Whisper model through Qualcomm AI Hub, point the app at
the exported model, and it runs the ASR step on the Hexagon NPU via the
Qualcomm AI Engine Direct (QNN) execution provider.

---

## Setup — development machine

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --engine cpu_dev
```

This uses `faster-whisper` (a CPU-friendly Whisper implementation) so you can
develop and test the full pipeline on a laptop that isn't Snapdragon-powered.

## Setup — Snapdragon-powered HP PC (production target)

```powershell
# NOTE: only 64-bit (AMDx64) Python is supported for qai_hub_models on
# Windows on Snapdragon — do not use the Windows ARM64 Python build.
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-snapdragon.txt

# Configure your Qualcomm AI Hub API token (free account at aihub.qualcomm.com)
qai-hub configure --api_token YOUR_API_TOKEN

# Export & quantize Whisper Small for the Snapdragon X Elite NPU
python scripts/export_whisper_qnn.py --device "Snapdragon X Elite CRD" --out models/whisper_small_quantized

# Run the app on the NPU engine
python main.py --engine qnn --model-dir models/whisper_small_quantized
```

## Benchmarking (for your Challenge submission)

```bash
python scripts/benchmark.py --model-dir models/whisper_small_quantized --sample audio_samples/sample_hi_en.wav
```

Prints a side-by-side latency / real-time-factor comparison between the CPU
fallback engine and the NPU (QNN) engine — this is the evidence to cite under
"Technical Implementation" in your proposal and pitch.

## Configuration

All runtime behavior lives in `config.yaml` — sample rate, chunk length,
language mode, translation on/off, overlay appearance, and transcript export
formats. See the comments in that file for details.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

The tests cover the pure-logic pieces (chunk-window math, transcript
formatting) that don't require a microphone, model weights, or Snapdragon
hardware, so they run anywhere.

## Notes on the ASR engine abstraction

`src/asr_engine.py` defines a common `ASREngine` interface with two
implementations:

- `CPUWhisperEngine` — `faster-whisper` on CPU, for development and CI.
- `QNNWhisperEngine` — loads the model exported by
  `scripts/export_whisper_qnn.py` and runs it through the Qualcomm AI Engine
  Direct (QNN) execution provider for ONNX Runtime, so inference executes on
  the Hexagon NPU.

The exact class/method names exposed by `qai_hub_models` can change between
releases, so `QNNWhisperEngine` is written against the documented pattern
(`Model.from_pretrained()` + a model-provided `App` wrapper with a
`.transcribe()`/`.predict()` method) and includes fallback error handling
that prints the command to run `python -m qai_hub_models.models.<model>.demo`
so you can confirm the current API on your machine before wiring it in fully.
