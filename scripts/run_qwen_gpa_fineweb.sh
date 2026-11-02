#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${GPA_PYTHON:-python}"
source_model="${MODEL:-Qwen/Qwen3-0.6B}"
model_tag="${source_model##*/}"
converted="${CONVERTED:-runs/${model_tag}_gpa_base}"
output="${OUTPUT:-runs/${model_tag}_gpa_fineweb}"
if [[ -z "${RESUME:-}" && ! -f "$converted/config.json" ]]; then
  conversion_options=()
  [[ -n "${SLOTS:-}" ]] && conversion_options+=(--slots "$SLOTS")
  [[ -n "${LAST_SLOTS:-}" ]] && conversion_options+=(--last-slots "$LAST_SLOTS")
  [[ "${KEEP_ROPE:-0}" == 1 ]] && conversion_options+=(--keep-rope)
  [[ "${QUADRATIC_FEATURES:-0}" == 1 ]] && conversion_options+=(--quadratic-features)
  "$python_bin" scripts/convert_qwen_to_gpa.py --model "$source_model" --output "$converted" \
      --backend fla --execution staged "${conversion_options[@]}"
fi
options=()
[[ -n "${DATA_FILES:-}" ]] && options+=(--data-files "$DATA_FILES")
[[ -n "${DEEPSPEED:-}" ]] && options+=(--deepspeed "$DEEPSPEED")
[[ -n "${RESUME:-}" ]] && options+=(--resume "$RESUME")
exec "$python_bin" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-1}" \
  scripts/train_gpa_fineweb.py --model "$converted" --output "$output" \
  --length "${LENGTH:-8192}" --microbatch "${MICROBATCH:-1}" \
  --tokens-per-update "${TOKENS_PER_UPDATE:-524288}" --max-steps "${MAX_STEPS:-10000}" \
  --lr "${LR:-1e-4}" --tuning lora --lora-rank "${LORA_RANK:-32}" "${options[@]}" "$@"
