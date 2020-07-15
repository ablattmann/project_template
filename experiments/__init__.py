from experiments.experiment import Experiment


__experiments__ = {
    "test": Experiment,
}


def select_experiment(config,dirs, device):
    experiment = config["general"]["experiment"]
    project_name = config["general"]["project_name"]
    if experiment not in __experiments__:
        raise NotImplementedError("No such experiment!")
    if config["general"]["restart"]:
        print(f"Restarting experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    else:
        print(f"Running new experiment \"{project_name}\" of type \"{experiment}\". Device: {device}")
    return __experiments__[experiment](config, dirs, device)
