# AI Learning Guide — TTS Preparation

Prepare spoken-word audio from the AI learning curriculum using the same pattern as the ham radio and service provider study guides.

## Current Decision Point

No audio generation should run until Jeremy approves the approach.

This folder only prepares:

- pronunciation dictionary
- preprocessing scripts
- chunking workflow
- proposed generation pipeline

## Voice Settings

| Setting | Value |
|---|---|
| Engine | Kokoro via Open Speech |
| Voice | `will` |
| Speed | 1.1x target, if supported by the active script |
| Format | MP3 |
| Bitrate | 128kbps MP3 |
| Max chunk | ~2000 chars per TTS call |

## Pronunciation Dictionary

See [`pronunciation.txt`](pronunciation.txt).

Important rule: **AI must be spoken as “A.I.”**, not “aye.”

Examples:

```text
AI|A.I.
LLM|L.L.M.
RAG|R.A.G.
API|A.P.I.
JSON|Jason
FastAPI|Fast A.P.I.
OpenAI|Open A.I.
PydanticAI|Pydantic A.I.
CrewAI|Crew A.I.
```

The dictionary is intentionally case-sensitive. Put longer/compound terms before shorter terms so `OpenAI` becomes `Open A.I.` before the generic `AI` rule runs.

## Proposed Input Files

The curriculum currently lives at repo root:

- `README.md`
- `month-01-foundations-of-ai.md`
- `month-02-the-transformer-era.md`
- `month-03-tokens-and-embeddings.md`
- `month-04-training-llms.md`
- `month-05-finetuning-and-alignment.md`
- `month-06-hardware-and-compute.md`
- `month-07-quantization.md`
- `month-08-inference-engines.md`
- `month-09-throughput-optimization.md`
- `month-10-distributed-training.md`
- `month-11-serving-and-autoscaling.md`
- `month-12-ecosystem-and-future.md`
- `parsed-links-for-input.md` should stay reference-only unless Jeremy wants a source-notes audio appendix.

## Proposed Output Layout

Recommended:

```text
tts/audio/
  00-overview.mp3
  month-01-foundations-of-ai.mp3
  month-02-the-transformer-era.mp3
  ...
  month-12-ecosystem-and-future.mp3
```

Keep generated MP3s out of the first commit unless we explicitly decide to version audio in Git.

## Pipeline

### 1. Convert Markdown to spoken text

For the first pass, use light markdown cleanup only:

- remove raw links unless they are part of the lesson
- remove table syntax
- remove code fences
- convert headings to spoken transitions
- keep bullets as short spoken list items

### 2. Apply pronunciation dictionary

```bash
python3 tts/scripts/apply-pronunciation.py input.txt tts/pronunciation.txt > pronounced.txt
```

### 3. Chunk text

```bash
python3 tts/scripts/chunk-text.py pronounced.txt --max-chars 2000 --output-dir /tmp/ai-learning-tts --prefix month-01
```

### 4. Generate audio only after approval

Do **not** run this until Jeremy approves:

```bash
python3 ~/.openclaw/skills/voice/scripts/tts-kokoro \
  "$(cat /tmp/ai-learning-tts/month-01_001.txt)" \
  --voice will --format mp3 \
  --output /tmp/ai-learning-tts/month-01_001.mp3
```

### 5. Concatenate chunks

```bash
ls /tmp/ai-learning-tts/month-01_*.mp3 | sort | sed "s|.*|file '&'|" > /tmp/ai-learning-tts/concat.txt
ffmpeg -y -f concat -safe 0 -i /tmp/ai-learning-tts/concat.txt -c:a libmp3lame -b:a 128k tts/audio/month-01-foundations-of-ai.mp3
```

## Validation Before Audio

Before generating audio, run pronunciation samples through preprocessing:

```bash
cat > /tmp/ai-tts-sample.txt <<'SAMPLE'
AI engineers build LLM systems with RAG, APIs, JSON outputs, FastAPI, OpenAI, PydanticAI, and MCP.
SAMPLE

python3 tts/scripts/apply-pronunciation.py /tmp/ai-tts-sample.txt tts/pronunciation.txt
```

Expected style:

```text
A.I. engineers build L.L.M. systems with R.A.G., A.P.I.s, Jason outputs, Fast A.P.I., Open A.I., Pydantic A.I., and M.C.P.
```

## Open Questions

1. Should `parsed-links-for-input.md` become an audio appendix, or remain source reference only?
2. Should generated MP3s be committed to Git, or stored as release/media artifacts?
3. Should every month be one MP3, or split into week-sized MP3s for easier listening?

My recommendation: **one MP3 per month**, no source appendix audio initially, and keep generated audio out of Git until file sizes are known.
