# %%
from src.config.config_atk import (
    configure_experiment,
    run_atk_experiments,
    run_atk_train_test,
)

if __name__ == "__main__":

    params, max_jobs = configure_experiment()

    training_fn = run_atk_train_test

    run_atk_experiments(params, max_jobs, training_fn)
