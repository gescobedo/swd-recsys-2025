from src.utils.helper import save_model
import numpy as np


def init_early_stopping_dict(criterion):
    best_validation_scores = {
        label: (-np.inf if crit.get("highest_is_best") else np.inf)
        for label, crit in criterion.items()
    }
    # setting up patience
    best_validation_scores["current_patience"] = 0
    return best_validation_scores


def test_early_stopping_condition(
    model,
    epoch,
    early_stopping_criteria,
    results_dict,
    best_validation_scores,
    result_path,
    is_verbose=False,
):
    for label, criterion in early_stopping_criteria.items():
        if (wu := criterion.get("warmup")) and epoch <= wu:
            break

        # Allow early stopping on non rank-based metrics
        validation_score = results_dict[criterion["metric"]]
        if "top_k" in criterion:
            validation_score = validation_score[criterion["top_k"]]

        better_model_found = False
        if "highest_is_best" in criterion:
            better_model_found = (
                validation_score >= best_validation_scores[label]
                and criterion["highest_is_best"]
                or validation_score <= best_validation_scores[label]
                and not criterion["highest_is_best"]
            )
        elif "closest_is_best" in criterion:
            old_diff = abs(criterion["value"] - best_validation_scores[label])
            new_diff = abs(criterion["value"] - validation_score)
            better_model_found = new_diff <= old_diff

        if better_model_found:
            if is_verbose:
                print("Better model found!")
                if top_k := criterion.get("top_k"):
                    print(f'{criterion["metric"]}@{top_k}={validation_score:.4f}\n')
                else:
                    print(f'{criterion["metric"]}={validation_score:.4f}\n')
            save_model(model, result_path, "best_model_" + label)
            best_validation_scores[label] = validation_score

        stop_training = False
        if "patience" in criterion:
            current_patience = best_validation_scores["current_patience"]
            # print([current_patience, criterion["patience"]])
            if better_model_found == True:
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= criterion["patience"]:
                    # print([current_patience, criterion["patience"]])
                    stop_training = True
            best_validation_scores["current_patience"] = current_patience

    return best_validation_scores, stop_training
