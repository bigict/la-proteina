# input: DATA_PATH 环境变量、training_ae 配置、PDB 数据目录
# output: 运行 prepare_data 并在 processed/ 生成 .pt
# pos: tests/pdb_data，小规模预处理脚本(验证 pipeline 可跑通)
# If you update this file, update this header AND the folder README.
"""
最小化预处理脚本，便于快速验证修改后能否产出 .pt 文件。

用法：
  export DATA_PATH=/home/xxx/DATA        # 必须指向包含 protna_step2b/raw 的目录
  python script_utils/dataset_preprocess.py

脚本会：
  - 使用训练配置 training_ae + step2b 数据集
  - 将 NA 子集 fraction 设为 0.001，开启 overwrite & 重新聚类
  - 关闭 wandb/ckpt
  - 调用 datamodule.prepare_data()
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
        raise SystemExit("请先 export DATA_PATH 指向数据根目录，例如 /home/xxx/DATA")

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
    print("done. 检查 processed 目录是否生成 .pt")


if __name__ == "__main__":
    main()
