#!/bin/sh
#


CWD=`realpath -s $0`
CWD=`dirname ${CWD}`

cd ${CWD}
pushd ..

PYTHONPATH=. DATA_PATH=${DATA_PATH:-.} python script_utils/dataset_preprocess.py \
    dataset.datamodule.dataselector.min_length=50 \
    dataset.datamodule.dataselector.max_length=${sequence_max_length:-512} \
    $*

popd
