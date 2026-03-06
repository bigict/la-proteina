#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}
pushd ..

sequence_max_length=${sequence_max_length:-512}
pretrain_ckpt=${pretrain_ckpt:-"laproteina"}

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python proteinfoundation/partial_autoencoder/train.py \
    run_name_="${pretrain_ckpt}_release_ae_${sequence_max_length}" \
    hardware.ngpus_per_node_=auto \
    dataset.datamodule.batch_size=1 \
    dataset.datamodule.dataselector.max_length=${sequence_max_length} \
    pretrain_ckpt_path=checkpoints_${pretrain_ckpt}/AE1_ucond_512.ckpt \
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
