#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` [-h] -m {model_ckpt_file} -- [infer_opt ...]"
  echo "positional arguments:"
  echo "    infer_opt  inference option."
  echo "               see configs/inference_ae.yaml for further help."
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -m MODEL_CKPT, --model_ckpt MODEL_CKPT"
  echo "               model checkpoint file."
  exit $1
}

ARGS=$(getopt -o "m:h" -l "model_ckpt:,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-m | --model_ckpt) model_ckpt="$2"; shift 2;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

if [ -z ${model_ckpt} ]; then
  help 1
fi

pushd ..

sequence_max_length=${sequence_max_length:-512}
dataset=${dataset:-"pdb/pdb_train_ucond"}

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/partial_autoencoder/inference.py \
    --config_name "inference_ae" \
    ckpt_file="${model_ckpt}" \
    +dataset=${dataset} \
    dataset.datamodule.batch_size=1 \
    dataset.datamodule.dataselector.max_length=${sequence_max_length} \
    $*

popd
