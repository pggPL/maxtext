#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
MAXTEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export MAXTEXT_DIR
export PYTHONPATH="${MAXTEXT_DIR}/src:${PYTHONPATH:-}"
export DECOUPLE_GCLOUD=TRUE
export NVTE_JAX_ENFORCE_V2_GROUPED_GEMM=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

GPUS="${GPUS:-4}"
case "$GPUS" in
  2) EP=2; FSDP=1 ;;
  4) EP=2; FSDP=2 ;;
  *) echo 'Use GPUS=2 or GPUS=4; TE EP requires at least two processes.' >&2; exit 2 ;;
esac
LAYERS="${LAYERS:-4}"
if ! [[ "$LAYERS" =~ ^[0-9]+$ ]] || (( LAYERS < 2 || LAYERS > 93 )); then
  echo 'LAYERS must be between 2 and 93; layer 0 alone does not exercise MoE.' >&2
  exit 2
fi
FULL_ATTN_LAYERS="${FULL_ATTN_LAYERS:-}"
if [[ -z "$FULL_ATTN_LAYERS" ]]; then
  FULL_ATTN_LAYERS='['
  for (( layer=3; layer<LAYERS; layer+=4 )); do FULL_ATTN_LAYERS+="$layer,"; done
  if (( LAYERS == 93 )); then FULL_ATTN_LAYERS+='92,'; fi
  FULL_ATTN_LAYERS="${FULL_ATTN_LAYERS%,}]"
fi
OUTPUT="${OUTPUT:-${MAXTEXT_DIR}/outputs/kimi_te_exact_$(date +%Y%m%d_%H%M%S)}"
EXTRA="override_model_config=true te_moe_block=true
base_num_decoder_layers=$LAYERS full_attn_layers=$FULL_ATTN_LAYERS num_experts=${EXPERTS:-32}
quantization= te_gmm_quantization=${TE_GMM_QUANTIZATION:-te_no_quant}
sparse_matmul=true megablox=false prefuse_moe_weights=false use_ring_of_experts=false
ici_expert_parallelism=$EP ici_fsdp_parallelism=$FSDP ici_data_parallelism=1
dcn_expert_parallelism=1 dcn_fsdp_parallelism=1 dcn_data_parallelism=1
scan_layers=${SCAN_LAYERS:-false} packing=false ragged_buffer_factor=0.0
max_target_length=${MAX_TARGET_LENGTH:-1024} weight_dtype=bfloat16 mu_dtype=bfloat16
use_iota_embed=false enable_checkpointing=false dataset_type=synthetic
jax_cache_dir=$OUTPUT/jax_cache ${EXTRA_MAXTEXT_ARGS:-}"
EXTRA="${EXTRA//$'\n'/ }"
CMD=(test-maxtext.sh --multiprocess -n 1 -b "${BATCH_PER_GPU:-1}"
  --model-name=kimi-k3 --attn-type=dot_product --remat-policy=full
  --steps="${STEPS:-20}" --data-parallel=1 --fsdp=1 --tensor-parallel=1
  --tensor-sequence-parallel=1 --mem-fraction 0.85 --output "$OUTPUT" -a "$EXTRA")
printf 'Kimi-K3 + TE: full-width router, two-branch SiTU. EP=%s FSDP=%s\n' "$EP" "$FSDP"
if [[ "${MOE_PARITY:-0}" == 1 ]]; then
  CMD=(python3 "$MAXTEXT_DIR/tests/end_to_end/gpu/kimi/check_te_exact_moe.py")
fi
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$OUTPUT/logs"
printf '%s\n' "$EXTRA" > "$OUTPUT/overrides.txt"
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for (( rank=0; rank<GPUS; rank++ )); do
  (
    export CUDA_VISIBLE_DEVICES="$rank" SLURM_STEP_GPUS="$rank"
    export SLURM_NTASKS="$GPUS" SLURM_NPROCS="$GPUS" SLURM_STEP_NUM_TASKS="$GPUS"
    export SLURM_PROCID="$rank" SLURM_LOCALID="$rank"
    unset OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_LOCAL_RANK
    export JAX_COORDINATOR_IP=127.0.0.1 JAX_COORDINATOR_PORT="${COORDINATOR_PORT:-21666}"
    export NNODES="$GPUS" NODE_RANK="$rank"
    export JAX_COORDINATOR_ADDRESS="127.0.0.1:$JAX_COORDINATOR_PORT"
    export MAXTEXT_JAX_COORDINATOR_ADDRESS="$JAX_COORDINATOR_ADDRESS"
    export JAX_PROCESS_COUNT="$GPUS" JAX_PROCESS_INDEX="$rank" JAX_LOCAL_DEVICE_IDS=0
    if [[ "${MOE_PARITY:-0}" != 1 ]]; then
      CMD[-1]="${CMD[-1]} metrics_file=$OUTPUT/metrics_rank_$rank.jsonl"
    fi
    exec "${CMD[@]}"
  ) > "$OUTPUT/logs/proc_$rank.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do if ! wait "$pid"; then FAILED=1; cleanup; fi; done
printf 'exit=%s\n' "$FAILED" > "$OUTPUT/status.txt"
exit "$FAILED"
