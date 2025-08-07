input_options = {
    "experiment_type": {
        "type": str,
        "required": True,
        "choices": ["standard", "up_sample"],
    },
    "dataset": {
        "type": str,
        "required": False,
        "default": "ml-1m",
        "help": "The dataset to train / run the models on.",
    },
    "algorithm": {
        "type": str,
        "required": False,
        "default": "multivae",
        "help": "The dataset to train / run the models on.",
    },
    "gpus": {
        "type": str,
        "required": False,
        "default": "",
        "help": "The gpus to run the models on, use e.g., '0,2' to run on GPU '0' and '2'",
    },
    "n_parallel": {
        "type": int,
        "required": False,
        "default": 1,
        "help": "The number of processes that should be run on each device",
    },
    "store_best": {
        "type": bool,
        "required": False,
        "default": False,
        "help": "Whether the best models found for each run should be stored, "
        "i.e., whether early stopping should be performed.",
    },
    "store_every": {
        "type": int,
        "required": False,
        "default": 0,
        "choices": range(0, 100),
        "help": "After which number of epochs the model should be stored, 0 to deactivate this feature",
    },
    "config": {
        "type": str,
        "required": True,
        "help": "The config file to use for running an experiment",
    },
    "atk_config": {
        "type": str,
        "required": True,
        "help": "The attacker config file to use for running an experiment (only in case of executing train+atk)",
    },
    "oracle_config": {
        "type": str,
        "required": True,
        "help": "The oracle config file to use for running an experiment ",
    },
    "split": {
        "type": str,
        "required": False,
        "default": "test",
        "choices": ["train", "val", "test"],
        "help": "The split to use.",
    },
    "use_tensorboard": {
        "type": bool,
        "required": False,
        "default": False,
        "help": "Whether additional information should be logged via tensorboard",
    },
    "experiment": {
        "type": str,
        "required": False,
        "default": None,
        "help": "The path to an experiment, i.e., collection of multiple runs, "
        "where each one should be validated",
    },
    "run": {
        "type": str,
        "required": False,
        "default": None,
        "help": "The path to a run that should be validated.",
    },
    "model_pattern": {
        "type": str,
        "required": False,
        "help": "If specified, only models that match this pattern are considered. "
        "(glob syntax is used)",
    },
    "perform_undersampling": {
        "type": bool,
        "required": False,
        "default": False,
        "help": "Whether to perform undersampling of majority group than oversampling minority group. "
        "This option leads to ignore the option 'oversample_ratio' (not available in imblearn).",
    },
    "results_dir": {
        "type": str,
        "required": False,
        "default": None,
        "help": "Include a custom dir to save your results",
    },
}
