# MiroThinker-v1.0-30B-FP8 with vLLM
#
# Build:
#   docker build -t mirothinker-fp8 .
#
# Run (TP=2, recommended):
#   docker run --gpus all -p 8000:8000 mirothinker-fp8
#
# Run (single GPU, not recommended):
#   docker run --gpus '"device=0"' -p 8000:8000 -e SINGLE_GPU=1 mirothinker-fp8

FROM vllm/vllm-openai:v0.11.2

ENV MODEL_ID="Doradus/MiroThinker-v1.0-30B-FP8"
ENV HOST="0.0.0.0"
ENV PORT="8000"

# Default: TP=2 for 2x GPUs (recommended)
ENV TENSOR_PARALLEL_SIZE="2"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTILIZATION="0.90"

# For single GPU fallback (set SINGLE_GPU=1)
ENV SINGLE_GPU=""

EXPOSE 8000

CMD if [ -n "$SINGLE_GPU" ]; then \
      echo "WARNING: Single GPU mode - limited context and throughput" && \
      python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_ID \
        --host $HOST \
        --port $PORT \
        --tensor-parallel-size 1 \
        --max-model-len 2048 \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --trust-remote-code; \
    else \
      python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_ID \
        --host $HOST \
        --port $PORT \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --trust-remote-code \
        --enable-chunked-prefill; \
    fi
