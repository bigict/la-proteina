#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` [-a -h] -t {ucond,motif} -- [train_opt ...]"
  echo "positional arguments:"
  echo "    train_opt  train option."
  echo "               see configs/train_ae.yaml for further help."
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -t TRAIN_MODE, --train_mode TRAIN_MODE {ucond,motif}"
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
elif [ x"${train_type}" == x"motif_idx" ]; then
    dataset="pdb/pdb_train_motif_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
elif [ x"${train_type}" == x"motif_uidx" ]; then
    dataset="pdb/pdb_train_motif_aa"
    autoencoder_ckpt_path="AE3_motif.ckpt"
else
    help 1;
fi

pushd ..

sequence_max_length=${sequence_max_length:-512}
pretrain_ckpt=${pretrain_ckpt:-"laproteina"}

pretrain_ckpt_path=${pretrain_ckpt}
if [ ${add_sequence_max_length_to_pretrain_ckpt} ]; then
  pretrain_ckpt_path=${pretrain_ckpt}_${sequence_max_length}
fi

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/partial_autoencoder/train.py \
    run_name_="${pretrain_ckpt}_release_ae_${sequence_max_length}" \
    hardware.ngpus_per_node_=auto \
    dataset=${dataset} \
    dataset.datamodule.batch_size=1 \
    dataset.datamodule.dataselector.max_length=${sequence_max_length} \
    pretrain_ckpt_path=checkpoints_${pretrain_ckpt_path}/${autoencoder_ckpt_path} \
    nn_ae.encoder.feats_seq="[chain_break_per_res, x1_aatype, x1_a37coors_nm, x1_a37coors_nm_rel, x1_bb_angles, x1_sidechain_angles]" \
    nn_ae.encoder.feats_pair_repr="[rel_seq_sep, x1_bb_pair_dists_nm, x1_bb_pair_orientation]" \
    opt.max_epochs=100 \
    opt.accumulate_grad_batches=4 \
    log.log_wandb=true \
    log.wandb_project=proteina_ae_na \
    $*

###############################
#    dataset.datamodule.sampling_mode=random \
#    dataset.datamodule.datasplitter.split_type=random \
###############################
popd
