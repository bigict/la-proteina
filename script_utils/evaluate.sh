#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` [-h] -t {ucond,motif_idx,motif_uidx} -- [eval_opt ...]"
  echo "positional arguments:"
  echo "    eval_opt  evaluate option."
  echo "               see configs/inference_*.yaml for further help."
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -t INFERENCE_MODE, --inference_mode INFERENCE_MODE {ucond,motif_idx,motif_uidx}"
  echo "               type of evaluate mode. (default: ucond)"
  exit $1
}

eval_type="ucond"

ARGS=$(getopt -o "t:h" -l "inference_mode:,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-t | --inference_mode) eval_type="$2"; shift 2;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

if [ x"${eval_type}" == x"ucond" ]; then
    config_name="ucond_tri"
    autoencoder_ckpt_path="AE1_ucond_512.ckpt"
elif [ x"${eval_type}" == x"motif_idx" ]; then
    config_name="motif_idx_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
elif [ x"${eval_type}" == x"motif_uidx" ]; then
    config_name="motif_uidx_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
else
    help 1;
fi

pushd ..

pretrain_ckpt=${pretrain_ckpt:-"laproteina"}

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/evaluate.py \
    --config_name "inference_${config_name}" \
    run_name_=${pretrain_ckpt}_${config_name} \
    ckpt_path=checkpoints_${pretrain_ckpt} \
    autoencoder_ckpt_path=checkpoints_${pretrain_ckpt}/${autoencoder_ckpt_path} \
    $*

popd
