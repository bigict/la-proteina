#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` [-a -h] -t {ucond,motif_idx,motif_uidx} -- [train_opt ...]"
  echo "positional arguments:"
  echo "    train_opt  train option."
  echo "               see configs/training_local_latents.yaml for further help."
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -t TRAIN_MODE, --train_mode TRAIN_MODE {ucond,motif_idx,motif_uidx}"
  echo "               type of train mode. (default: ucond)"
  echo "    -a, --add_sequence_max_length_to_pretrain_ckpt"
  echo "               add sequence_max_length to pretrain_ckpt. (default: false)"
  exit $1
}

train_type="ucond"
add_sequence_max_length_to_pretrain_ckpt=0

ARGS=$(getopt -o "t:ah" -l "train_mode:,add_sequence_max_length_to_pretrain_ckpt,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-t | --train_mode) train_type="$2"; shift 2;;
    (-a | --add_sequence_max_length_to_pretrain_ckpt) add_sequence_max_length_to_pretrain_ckpt=1; shift 1;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

if [ x"${train_type}" == x"ucond" ]; then
    dataset="pdb/pdb_train_ucond"
    autoencoder_ckpt_path="AE1_ucond_512.ckpt"
    nn="local_latents_score_nn_160M"
    nn_ckpt_path="LD2_ucond_tri_512.ckpt"
elif [ x"${train_type}" == x"motif_idx" ]; then
    dataset="pdb/pdb_train_motif_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
    nn="local_latents_score_nn_160M_motif_idx_aa"
    nn_ckpt_path="LD4_motif_idx_aa.ckpt"
elif [ x"${train_type}" == x"motif_uidx" ]; then
    dataset="pdb/pdb_train_motif_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
    nn="local_latents_score_nn_160M_motif_uidx"
    nn_ckpt_path="LD6_motif_uidx_aa.ckpt"
else
    help 1;
fi

pushd ..

sequence_max_length=${sequence_max_length:-512}
ae_pretrain_ckpt=${ae_pretrain_ckpt:-"laproteina"}
nn_pretrain_ckpt=${nn_pretrain_ckpt:-"laproteina"}

ae_pretrain_ckpt_path=${ae_pretrain_ckpt}
if [ ${add_sequence_max_length_to_pretrain_ckpt} ]; then
  ae_pretrain_ckpt_path=${ae_pretrain_ckpt}_${sequence_max_length}
fi
nn_pretrain_ckpt_path=${nn_pretrain_ckpt}
if [ ${add_sequence_max_length_to_pretrain_ckpt} ]; then
  nn_pretrain_ckpt_path=${nn_pretrain_ckpt}_${sequence_max_length}
fi

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/train.py \
    run_name_="${nn_pretrain_ckpt}_release_diffusion_${sequence_max_length}" \
    hardware.ngpus_per_node_=auto \
    dataset=${dataset} \
    dataset.datamodule.batch_size=2 \
    dataset.datamodule.dataselector.max_length=${sequence_max_length} \
    nn=${nn} \
    +nn.use_residue_type_x=true \
    autoencoder_ckpt_path=checkpoints_${ae_pretrain_ckpt_path}/${autoencoder_ckpt_path} \
    pretrain_ckpt_path=checkpoints_${nn_pretrain_ckpt_path}/${nn_ckpt_path} \
    log.log_wandb=true \
    log.wandb_project=proteina_fm_na \
    +is_cluster_run=true \
    $*

##################################
#    dataset.datamodule.sampling_mode=random \
#    dataset.datamodule.datasplitter.split_type=random \
##################################
popd
