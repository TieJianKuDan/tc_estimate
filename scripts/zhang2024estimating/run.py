import sys

sys.path.append("./")

from core.zhang2024estimating.model import TCIntensityNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import HURSATB1PL


def main():
    cc = ClassicConf(
        dm_class=HURSATB1PL,
        model_class=TCIntensityNetPL,
        config_dir="scripts/zhang2024estimating/config.yaml",
        log_dir="./logs/",
        log_name="zhang2024estimating",
        ckp_dir="ckps/zhang2024estimating"
    )
    cc.train()

if __name__ == "__main__":
    main()
    