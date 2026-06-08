# Architecture

The system is split into independently testable services:

- `asr`: speech-to-text with Whisper-compatible Hugging Face models.
- `emotion`: text emotion classification.
- `dialogue`: prompt construction, response parsing, and causal LM generation.
- `generation`: auxiliary emotional GPT-2 generation.
- `interest`: lightweight keyword-based scoring.
- `tts`: speech synthesis through Coqui TTS as an external dependency.

The package follows a lazy-loading rule: importing modules should not download
models, allocate GPU memory, or start audio devices. Heavy resources are loaded
only when a service class is instantiated.

