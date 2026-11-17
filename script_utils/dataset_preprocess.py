# input: DATA_PATH environment variable, training_ae config, PDB data directory
# output: runs prepare_data and writes .pt files under processed/
# pos: tests/pdb_data, a small-scale preprocessing script (checks the pipeline runs)
# If you update this file, update this header AND the folder README.
"""
Minimal preprocessing script, for a quick check that a change still produces .pt files.

Usage:
  export DATA_PATH=/home/xxx/DATA        # must point at the directory holding protna_step2b/raw
  python script_utils/dataset_preprocess.py

The script:
  - uses the training_ae config with the step2b dataset
  - sets the NA subset fraction to 0.001, enables overwrite and re-clustering
  - disables wandb/ckpt
  - calls datamodule.prepare_data()
"""

import os
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

from proteinfoundation.partial_autoencoder.train import load_data_module


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="training_ae",
)
def main(cfg):
    data_path = os.environ.get("DATA_PATH")
    if not data_path:
        raise SystemExit("export DATA_PATH to the data root first, e.g. /home/xxx/DATA")

    # config_dir = str(Path(__file__).resolve().parents[2] / "proteinfoundation" / "configs")
    # with initialize_config_dir(version_base=None, config_dir=config_dir):
    #     cfg = compose(
    #         config_name="training_ae",
    #         overrides=[
    #             "dataset=pdb/deeppbs_train_ucond_debug",
    #             "dataset.datamodule.overwrite=true",
    #             "dataset.datamodule.datasplitter.overwrite_sequence_clusters=true",
    #             "log.log_wandb=false",
    #             "log.checkpoint=false",
    #             "hardware.ncpus_per_task_train_=4",
    #         ],
    #     )

    cfg_data, dm = load_data_module(cfg, is_cluster_run=False)
    dm.prepare_data()
    dm.setup()
    print("done. check the processed directory for the generated .pt files")


if __name__ == "__main__":
    main()
