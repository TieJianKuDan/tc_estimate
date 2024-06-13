import sys

sys.path.append("./")

from core.zhang2021tropical.tcic import TCICPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import HURSATB1PL


def main():
    cc = ClassicConf(
        dm_class=HURSATB1PL,
        model_class=TCICPL,
        config_dir="scripts/zhang2021tropical/config.yaml",
        log_dir="./logs/zhang2021tropical",
        log_name="tcice",
        ckp_dir="ckps/zhang2021tropical/tcice"
    )
    cc.train()

if __name__ == "__main__":
    main()
    