#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage:
  run_complex_pwm.sh predict <pdb_or_dir> <ae.ckpt> <output_dir> [device]
  run_complex_pwm.sh eval <input.txt> <prediction_dir> <output_dir>

Examples:
  bash script_utils/run_complex_pwm.sh predict \
    data/complex.pdb checkpoints/protein_dna_ae.ckpt results/pwm

  bash script_utils/run_complex_pwm.sh predict \
    data/gt_deeppbs_extracted checkpoints/protein_dna_ae.ckpt results/deeppbs_pwm

  bash script_utils/run_complex_pwm.sh eval \
    script_utils/eval_pwm_input_from_deeppbs.txt results/deeppbs_pwm results/eval
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command_name="$1"
shift

case "${command_name}" in
  predict)
    if [[ $# -lt 3 || $# -gt 4 ]]; then
      usage
      exit 2
    fi
    pdb_input="$1"
    checkpoint="$2"
    output_dir="$3"
    device="${4:-}"
    command_args=("${PYTHON_BIN}" "${SCRIPT_DIR}/predict_pwm_from_complex.py" \
      --input "${pdb_input}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${output_dir}")
    [[ -n "${device}" ]] && command_args+=(--device "${device}")
    "${command_args[@]}"
    ;;

  eval)
    if [[ $# -ne 3 ]]; then
      usage
      exit 2
    fi
    input_file="$1"
    prediction_dir="$2"
    output_dir="$3"
    command_args=("${PYTHON_BIN}" "${SCRIPT_DIR}/eval_pwm_from_complex.py" \
      "${input_file}" \
      --prediction-dir "${prediction_dir}" \
      --output-dir "${output_dir}")
    "${command_args[@]}"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
