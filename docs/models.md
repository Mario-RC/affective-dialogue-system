# Model Management

Model weights are intentionally excluded from Git.

Recommended storage:

- Dialogue model: Hugging Face `mario-rc/emotional-rlaif-dpo-gemma-2-2b-it`.
- Emotional GPT-2: Hugging Face `mario-rc/emotional-gpt2-medium`.
- Emotion classifiers:
  - `mario-rc/multilingual-emotional-classifier-xlm-roberta-base`
  - `mario-rc/multilingual-emotional-classifier-xlm-roberta-large`
- ASR: Hugging Face `openai/whisper-large-v3-turbo`.
- TTS: Coqui model catalog or a separate model artifact store.

If a model must be versioned with GitHub, use Git LFS in a separate model repo
or attach it to a release. Avoid committing training checkpoints to this source
repository.

