# %%
from src.config.config_exp import (
    run_experiments,
    configure_experiment,
    run_training,
    run_training_test,
    run_rec_training_test,
    run_test,
)


if __name__ == "__main__":

    params = configure_experiment()

    training_fn = run_rec_training_test
    test_fn = run_test
    run_experiments(params, 1, training_fn)

