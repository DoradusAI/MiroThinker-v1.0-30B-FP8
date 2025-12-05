# MiroThinker-v1.0-30B-FP8

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/68525b342230a897a65cc1c0/87mYQ_a-4jpnMkVR4hrgm.png" width="55%" alt="MiroThinker" />

  **FP8 quantized version of MiroThinker-30B for efficient inference**

  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Doradus/MiroThinker-v1.0-30B-FP8)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

## Download

**Model weights are hosted on HuggingFace:**

https://huggingface.co/Doradus/MiroThinker-v1.0-30B-FP8

This repository contains:
- Quantization scripts
- Docker deployment examples
- Documentation

## Quick Start

### Docker (Easiest)

```bash
# Download docker-compose.yml
wget https://huggingface.co/Doradus/MiroThinker-v1.0-30B-FP8/raw/main/docker/docker-compose.yml

# Run with 2 GPUs (recommended)
docker compose up

# Or single GPU (limited performance)
SINGLE_GPU=1 docker compose up
```

### vLLM

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model Doradus/MiroThinker-v1.0-30B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --trust-remote-code
```

## Hardware Requirements

| GPU Setup | Max Context | Performance | Notes |
|-----------|-------------|-------------|-------|
| 1x RTX 4090 (24GB) | OOM | N/A | Model too large |
| 1x RTX 5090 (32GB) | ~2K tokens | ~1-2 tok/s | Requires `--enforce-eager`, **not recommended** |
| **2x RTX 4090 TP=2** | ~16K tokens | ~60 tok/s | **Recommended consumer config** |
| **2x RTX 5090 TP=2** | ~32K tokens | ~80 tok/s | **Recommended consumer config** |
| 1x A100-80GB | ~131K tokens | ~60 tok/s | TP=1 possible |
| 1x H100-80GB | ~262K tokens | ~120 tok/s | Full context |

## Quantization

This model was quantized using [llmcompressor](https://github.com/vllm-project/llm-compressor):

```bash
python scripts/quantize_fp8.py
```

See [scripts/quantize_fp8.py](scripts/quantize_fp8.py) for the full script.

### Quantization Details

| Property | Value |
|----------|-------|
| Method | FP8 Dynamic (W8A8) |
| Weights | FP8 E4M3 (8-bit) |
| Activations | FP8 E4M3 (dynamic) |
| Original Size | ~60GB |
| Quantized Size | ~30GB |

## Original Model

Based on [miromind-ai/MiroThinker-v1.0-30B](https://huggingface.co/miromind-ai/MiroThinker-v1.0-30B).

MiroThinker is an open-source research agent with:
- 256K context window
- Up to 600 tool calls per task
- Strong performance on GAIA, BrowseComp, HLE-Text

See the [paper](https://arxiv.org/abs/2511.11793) for details.

## License

MIT License (inherited from original model)

## Citation

```bibtex
@article{miromind2025mirothinker,
  title={MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents},
  author={MiroMind Team},
  journal={arXiv preprint arXiv:2511.11793},
  year={2025}
}
```

## Acknowledgements

- [MiroMind AI](https://miromind.ai/) - Original MiroThinker model
- [Neural Magic / vLLM](https://github.com/vllm-project/llm-compressor) - llmcompressor
- [DoradusAI](https://doradusonline.com) - FP8 quantization
