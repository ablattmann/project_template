import argparse
from os import path, makedirs
from experiments import select_experiment
import torch
import yaml

def create_dir_structure(config):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["base_dir"],config["experiment"],subdir,config["project_name"]) for subdir in subdirs}
    return structure

def load_parameters(config_name, restart):
    with open(config_name,"r") as f:
        cdict = yaml.load(f,Loader=yaml.FullLoader)

    dir_structure = create_dir_structure(cdict["general"])
    saved_config = path.join(dir_structure["config"], "config.yaml")
    if restart:
        if path.isfile(saved_config):
            with open(saved_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    else:
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        if path.isfile(saved_config) and not cdict["general"]["debug"]:
            print(f"\033[93m" + "WARNING: Model has been started somewhen earlier: Resume training (y/n)?" + "\033[0m")
            while True:
                answer = input()
                if answer == "y" or answer == "yes":
                    with open(saved_config,"r") as f:
                        cdict = yaml.load(f, Loader=yaml.FullLoader)

                    restart = True
                    break
                elif answer == "n" or answer == "no":
                    with open(saved_config, "w") as f:
                        yaml.dump(cdict, f, default_flow_style=False)
                    break
                else:
                    print(f"\033[93m" + "Invalid answer! Try again!(y/n)" + "\033[0m")
        else:
            with open(saved_config, "w") as f:
                yaml.dump(cdict,f,default_flow_style=False)

    return cdict, dir_structure, restart


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/behavior_net.yaml",
                        help="Define config file")
    parser.add_argument("-r","--restart", type=bool, default=False,help="Whether training should be resumed.")
    parser.add_argument("--gpu",default=[0], type=int,
                        nargs="+",help="GPU to use.")
    parser.add_argument("-m","--mode",default="train",type=str,choices=["train","infer"],help="Whether to start in train or infer mode?")
    args = parser.parse_args()


    config, structure, restart = load_parameters(args.config, args.restart)
    config["general"]["restart"] = restart
    config["general"]["mode"] = args.mode

    if len(args.gpu) == 1:
        gpus = torch.device(
            f"cuda:{int(args.gpu[0])}"
            if torch.cuda.is_available() and int(args.gpu[0]) >= 0
            else "cpu"
        )
    else:
        gpus = [int(id) for id in args.gpu]

    experiment = select_experiment(config,structure, gpus)

    # start selected experiment
    mode = config["general"]["mode"]
    if  mode == "train":
        experiment.train()
    elif mode == "test":
        experiment.test()
    else:
        raise ValueError(f"\"mode\"-parameter should be either \"train\" or \"infer\" but is actually {mode}")