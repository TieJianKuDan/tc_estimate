import sys

sys.path.append("./")

from core.tan2022tropical.model import TCIntensityNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import HURSATB1PL


def main():
    cc = ClassicConf(
        dm_class=HURSATB1PL,
        model_class=TCIntensityNetPL,
        config_dir="scripts/tan2022tropical/config.yaml",
        log_dir="./logs/",
        log_name="tan2022tropical",
        ckp_dir="ckps/tan2022tropical"
    )
    cc.train()

if __name__ == "__main__":
    main()
    