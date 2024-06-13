import sys

sys.path.append("./")

from core.ours.model import TCIntensityNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import HURSATB1PL


def main():
    cc = ClassicConf(
        dm_class=HURSATB1PL,
        model_class=TCIntensityNetPL,
        config_dir="scripts/ours/config.yaml",
        log_dir="./logs/",
        log_name="ours",
        ckp_dir="ckps/ours"
    )
    cc.train()

if __name__ == "__main__":
    main()
    