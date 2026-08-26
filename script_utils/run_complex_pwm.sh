#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage:
  run_complex_pwm.sh predict <complex.pdb> <ae.ckpt> <output_prefix> [device]
  run_complex_pwm.sh eval <input.txt> <ae.ckpt> <output_dir> [pdb_root] [device]

Examples:
  bash script_utils/run_complex_pwm.sh predict \
    data/complex.pdb checkpoints/protein_dna_ae.ckpt results/complex_pwm

  bash script_utils/run_complex_pwm.sh eval \
    data/eval_input.txt checkpoints/protein_dna_ae.ckpt results/eval data/complexes
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
    pdb_path="$1"
    checkpoint="$2"
    output_prefix="$3"
    device="${4:-}"
    command_args=("${PYTHON_BIN}" "${SCRIPT_DIR}/predict_pwm_from_complex.py" \
      --pdb "${pdb_path}" \
      --checkpoint "${checkpoint}" \
      --output-prefix "${output_prefix}")
    [[ -n "${device}" ]] && command_args+=(--device "${device}")
    "${command_args[@]}"
    ;;

  eval)
    if [[ $# -lt 3 || $# -gt 5 ]]; then
      usage
      exit 2
    fi
    input_file="$1"
    checkpoint="$2"
    output_dir="$3"
    pdb_root="${4:-.}"
    device="${5:-}"
    command_args=("${PYTHON_BIN}" "${SCRIPT_DIR}/eval_pwm_from_complex.py" \
      "${input_file}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${output_dir}" \
      --pdb-root "${pdb_root}")
    [[ -n "${device}" ]] && command_args+=(--device "${device}")
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
