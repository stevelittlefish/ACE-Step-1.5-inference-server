#!/usr/bin/env bash
set -e

echo "==========================================="
echo "  ACE-Step 1.5"
echo "==========================================="
echo "Mode      : ${ACESTEP_MODE}"
echo "Python    : $(uv run python --version 2>&1)"
echo "PyTorch   : $(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

if uv run python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "CUDA      : $(uv run python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU       : $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "Memory    : $(uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.total_memory/1024**3:.1f} GB")')"
else
    echo "CUDA      : NOT AVAILABLE — running on CPU"
    echo "           (make sure you launched with --gpus all)"
fi
echo "==========================================="

# Build Gradio-only --init_service flags. The API server reads its model
# initialization settings directly from the environment.
INIT_ARGS=""
if [ "${ACESTEP_INIT_SERVICE:-true}" = "true" ]; then
    INIT_ARGS="--init_service true"
    [ -n "${ACESTEP_CONFIG_PATH:-}" ] &&
        INIT_ARGS="${INIT_ARGS} --config_path ${ACESTEP_CONFIG_PATH}"
    [ -n "${ACESTEP_LM_MODEL_PATH:-}" ] &&
        INIT_ARGS="${INIT_ARGS} --init_llm true --lm_model_path ${ACESTEP_LM_MODEL_PATH}"
    echo "Auto-init    : DiT=${ACESTEP_CONFIG_PATH:-auto}  LM=${ACESTEP_LM_MODEL_PATH:-none}"
fi

if [ "${ACESTEP_MODE}" = "api" ]; then
    echo "Starting REST API server on 0.0.0.0:${ACESTEP_API_PORT:-2766} ..."
    # ACESTEP_EXTRA_ARGS is intentionally word-split to support optional CLI flags.
    # shellcheck disable=SC2086
    exec uv run python -m acestep.api_server \
        --host "${ACESTEP_API_HOST:-0.0.0.0}" \
        --port "${ACESTEP_API_PORT:-2766}" \
        ${ACESTEP_EXTRA_ARGS:-}
else
    echo "Starting Gradio UI on 0.0.0.0:${GRADIO_PORT:-7860} ..."
    # INIT_ARGS and ACESTEP_EXTRA_ARGS are intentionally word-split CLI flags.
    # shellcheck disable=SC2086
    exec uv run python -m acestep.acestep_v15_pipeline \
        --server-name "${GRADIO_SERVER_NAME:-0.0.0.0}" \
        --port "${GRADIO_PORT:-7860}" \
        --backend "${ACESTEP_LLM_BACKEND:-pt}" \
        ${INIT_ARGS} \
        ${ACESTEP_EXTRA_ARGS:-}
fi
