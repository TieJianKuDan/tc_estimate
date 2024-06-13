import sys

sys.path.append("./")

from core.ma2024multiscale.model import TCIntensityNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import HURSATB1PL


def main():
    cc = ClassicConf(
        dm_class=HURSATB1PL,
        model_class=TCIntensityNetPL,
        config_dir="scripts/ma2024multiscale/config.yaml",
        log_dir="./logs/",
        log_name="ma2024multiscale",
        ckp_dir="ckps/ma2024multiscale"
    )
    cc.train()

if __name__ == "__main__":
    main()
    