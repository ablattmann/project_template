from abc import abstractmethod
import torch
import wandb
import os
from os import path
from glob import glob

WANDB_DISABLE_CODE = True

GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
ENDC = "\033[0m"

class Experiment:
    def __init__(self, config:dict, dirs: dict, device, **kwargs):
        self.parallel = isinstance(device, list)
        self.config = config
        if self.parallel:
            self.device = torch.device(
                f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
            )
            self.all_devices = device
            print("Running experiment on multiple gpus!")
        else:
            self.device = device
            self.all_devices = [device]
        self.dirs = dirs

        if self.config["general"]["mode"] == "train":
            wandb.init(
                dir=self.dirs["log"],
                name=self.config["general"]["project_name"],
                group=self.config["general"]["experiment"],
                sync_tensorboard=True,
            )

    def _load_ckpt(self, key, dir=None,name=None):
        if dir is None:
            dir = self.dirs["ckpt"]

        if name is None:
            if len(os.listdir(dir)) > 0:
                ckpts = glob(path.join(dir,"*.pth"))

                # load latest stored checkpoint
                ckpts = [ckpt for ckpt in ckpts if key in ckpt.split("/")[-1]]
                if len(ckpts) == 0:
                    print(
                        RED + f"*************No ckpt found****************" + ENDC
                    )
                    op_ckpt = mod_ckpt = None
                    return mod_ckpt, op_ckpt
                ckpts = {float(x.split("_")[-1].split(".")[0]): x for x in ckpts}

                ckpt = torch.load(
                    ckpts[max(list(ckpts.keys()))], map_location="cpu"
                )

                mod_ckpt = ckpt["model"] if "model" in ckpt else None
                op_ckpt = ckpt["optimizer"] if "optimizer" in ckpt else None
                if mod_ckpt is not None:
                    print(
                        GREEN
                        + f"*************Restored model with key {key} from checkpoint****************"
                        + ENDC
                    )
                else:
                    print(
                        RED
                        + f"*************No ckpt for model with key {key} found, not restoring...****************"
                        + ENDC
                    )

                if op_ckpt is not None:
                    print(
                        GREEN
                        + f"*************Restored optimizer with key {key} from checkpoint****************"
                        + ENDC
                    )
                else:
                    print(
                        RED
                        + f"*************No ckpt for optimizer with key {key} found, not restoring...****************"
                        + ENDC
                    )
            else:
                mod_ckpt = op_ckpt = None

            return mod_ckpt, op_ckpt

        else:
            ckpt_path = path.join(dir,name)
            if not path.isfile(ckpt_path):
                print(
                    RED
                    + f"*************No ckpt for model and optimizer found under {ckpt_path}, not restoring...****************"
                    + ENDC
                )
                mod_ckpt = op_ckpt = None
            else:
                if "epoch_ckpts" in ckpt_path:
                    mod_ckpt = torch.load(
                        ckpt_path, map_location="cpu"
                    )
                    op_path = ckpt_path.replace("model@","opt@")
                    op_ckpt = torch.load(op_path,map_location="cpu")
                    return mod_ckpt,op_ckpt

                ckpt = torch.load(ckpt_path, map_location="cpu")
                mod_ckpt = ckpt["model"] if "model" in ckpt else None
                op_ckpt = ckpt["optimizer"] if "optimizer" in ckpt else None

                if mod_ckpt is not None:
                    print(
                        GREEN
                        + f"*************Restored model under {ckpt_path} ****************"
                        + ENDC
                    )
                else:
                    print(
                        RED
                        + f"*************No ckpt for model found under {ckpt_path}, not restoring...****************"
                        + ENDC
                    )

                if op_ckpt is not None:
                    print(
                        GREEN
                        + f"*************Restored optimizer under {ckpt_path}****************"
                        + ENDC
                    )
                else:
                    print(
                        RED
                        + f"*************No ckpt for optimizer found under {ckpt_path}, not restoring...****************"
                        + ENDC
                    )

            return mod_ckpt,op_ckpt

    @abstractmethod
    def train(self):
        """
        Here, the experiment shall be run
        :return:
        """
        pass

    @abstractmethod
    def test(self):
        """
        Here the prediction shall be run
        :param ckpt_path: The path where the checkpoint file to load can be found
        :return:
        """
        pass
