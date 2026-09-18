# Model management

Install only the optional integrations you use. Model weights are loaded at
service construction and should be cached separately from source control.

## Configuration and downloads

Use `load_config("configs/default.yaml")` for typed settings. The CLI loads that
file only when passed `--config`. Exported `ADS_DEVICE`, `ADS_CACHE_DIR`, and
`ADS_SPEAKER_WAV` override their corresponding configuration fields; `.env` files
are not read automatically.

`create_dialogue_engine(config)` passes runtime settings to the dialogue model.
Other services accept their own model identifier, device, and, for Hugging Face
adapters, `cache_dir`. For example:

```python
from affective_dialogue_system.asr import WhisperASR
from affective_dialogue_system.config import load_config

config = load_config()
asr = WhisperASR(
    model_id=config.models.asr_model,
    device=config.runtime.device,
    torch_dtype=config.runtime.torch_dtype,
    cache_dir=config.runtime.cache_dir,
)
print(asr.transcribe("sample.wav", language="es").text)
```

File-based ASR requires FFmpeg on PATH. Transcriptions retain the full text,
including all chunks returned by the pipeline.

To pre-download a Hugging Face model explicitly:

```bash
python scripts/download_models.py --cache-dir /path/to/cache \
  mario-rc/emotional-rlaif-dpo-gemma-2-2b-it
```

With no model arguments, the script downloads the dialogue model, emotional GPT-2,
and both emotion classifiers. It does not download Whisper, moderation models, or
Coqui TTS. Supply their Hugging Face identifiers explicitly where applicable.

## Generation

Supported prompt templates are `gemma` and `llama3`; the selected model must match
the template. `trust_remote_code` defaults to `false`. CPU inference defaults to
float32 and CUDA to float16; an explicit `torch_dtype` may select `float16`,
`float32`, `bfloat16`, or `float64`, subject to model and hardware support.

The project uses the Transformers 4.x API. Unit tests validate adapter contracts
without loading weights; run an inference smoke test with your chosen model,
dtype, context length, and device before relying on the deployment.

## Speech synthesis

The `tts` extra uses Coqui TTS 0.22 and requires Python 3.10 or 3.11. The upstream
package declares its Python requirements on [PyPI](https://pypi.org/project/TTS/).
XTTS needs a speaker reference or an appropriate speaker configuration. See the
[XTTS API documentation](https://docs.coqui.ai/en/latest/models/xtts.html).

```python
from affective_dialogue_system.config import load_config
from affective_dialogue_system.tts import CoquiTTSService

config = load_config()
tts = CoquiTTSService(
    model_name=config.models.tts_model,
    device=config.runtime.device,
    speaker_wav=config.models.speaker_wav,
)
tts.synthesize_to_file("Hola, soy Ray.", "outputs/greeting.wav", language="es")
```

Set `ADS_SPEAKER_WAV` to an existing reference file before running that example.
Custom `model_path` and `config_path` must be supplied together. The wrapper does
not automatically accept model terms on the caller's behalf; complete any required
upstream model-access steps separately.

## Optional toxicity detectors

Enable `toxicity.detoxify.enabled` or `toxicity.llama_guard.enabled` in YAML after
installing `.[toxicity]`. Both are disabled by default. Model availability and access
requirements are independent of package installation.

Llama Guard 2 emits `safe` or `unsafe` plus S1–S11 category codes. The adapter maps
these into the project's categories and flags malformed verdicts. Taxonomy details
are in the [official model card](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B).
An optional model adds another detector; it does not establish complete safety or
clinical suitability.
