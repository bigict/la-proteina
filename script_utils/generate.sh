#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` [-c -h] -t {ucond,motif_idx,motif_uidx} -c -j [JOB_ID] -- [infer_opt ...]"
  echo "positional arguments:"
  echo "    infer_opt  inference option."
  echo "               see configs/inference_*.yaml for further help."
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -t INFERENCE_MODE, --inference_mode INFERENCE_MODE {ucond,motif_idx,motif_uidx}"
  echo "               type of inference mode. (default: ucond)"
  echo "    -c, --create_config create config if not exist"
  echo "    -j JOB_ID, --job_id JOB_ID"
  echo "               job id. (default: 0)"
  exit $1
}

infer_type="ucond"
job_id=0

ARGS=$(getopt -o "t:cj:h" -l "inference_mode:,create_config,job_id:,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-t | --inference_mode) infer_type="$2"; shift 2;;
    (-j | --job_id) job_id="$2"; shift 2;;
    (-c | --create_config) create_config=1; shift 1;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

if [ x"${infer_type}" == x"ucond" ]; then
    config_name="ucond_tri"
    autoencoder_ckpt_path="AE1_ucond_512.ckpt"
elif [ x"${infer_type}" == x"motif_idx" ]; then
    config_name="motif_idx_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
elif [ x"${infer_type}" == x"motif_uidx" ]; then
    config_name="motif_uidx_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
else
    help 1;
fi

pushd ..

pretrain_ckpt=${pretrain_ckpt:-"laproteina"}
config_name="inference_${config_name}"
if [ ${create_config} -eq 1 ]; then
  pushd configs
  if [ ! -f ${config_name}_${pretrain_ckpt}.yaml ]; then
    ln -s ${config_name}.yaml ${config_name}_${pretrain_ckpt}.yaml
  fi
  popd
fi

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/generate.py \
    --config_name "${config_name}_${pretrain_ckpt}" \
    --job_id ${job_id} \
    run_name_=${pretrain_ckpt}_${config_name} \
    +nn.use_residue_type_x=true \
    ckpt_path=checkpoints_${pretrain_ckpt} \
    autoencoder_ckpt_path=checkpoints_${pretrain_ckpt}/${autoencoder_ckpt_path} \
    $*

# +generation.apply_residue_type_filter=true

popd
