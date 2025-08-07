from src.trainer.BaseTrainer import BaseTrainer
from src.data.BaseDataset import BaseDataset
from src.utils.helper import yaml_dump, save_model
from src.recsys_models.BaseRecModel import BaseRecModel, PairWiseRecModel
from src.nn_modules.mlp_attacker import MLPAttackerBaseModel
from torch.utils.data import DataLoader
import torch
from src.evaluation.classification import calculate_atk_metrics

from collections import defaultdict
from typing import Tuple, Dict
from src.utils.train_utils import (
    add_to_dict,
    scale_dict_items,
)
from src.utils.early_stopping import (
    init_early_stopping_dict,
    test_early_stopping_condition,
)
from src.utils.wandb_utils import wandb_write_to_run
import os
from tqdm import trange, tqdm
import wandb


class AttackNetTrainer(BaseTrainer):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        # self.atk_index = config.get("atk_index", 0)
        if self.use_wandb:
            entity = config["rec_model_config"]["entity"]
            project = config["rec_model_config"]["project"]
            sweep = config["rec_model_config"]["sweep"]
            run_id = config["rec_model_config"]["run_id"]
            self.wandb_runname = f"{entity}/{project}/{sweep}/{run_id}"

    def run_epoch(
        self,
        rec_model: BaseRecModel,
        model: MLPAttackerBaseModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "atk/train",
        training: bool = True,
    ):
        rec_model.eval()
        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for indices, model_input, user_features in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(indices)
            sample_count += n_samples
            # targets = targets.to(device)
            user_features = user_features[0].to(device)
            # TODO: dirty  fix should be transfered to other class for Pairwise models
            if isinstance(rec_model, PairWiseRecModel):
                model_input = indices.to(device)
            else:
                model_input = model_input.to(device)
            model_input = rec_model.encode_user(model_input).detach()
            logits = model(model_input)
            loss, loss_dict = model.calc_loss(logits, user_features)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not training:
                model_logits.append(logits.detach().cpu())
                model_targets.append(user_features.detach().cpu())

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        if not training:
            model_logits = torch.cat(model_logits)
            model_targets = torch.cat(model_targets)
        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss",
            scale_dict_items(train_loss_dict, 1 / sample_count),
            epoch,
        )
        # print([f"{phase}", epoch_avg_loss])
        return epoch_avg_loss, model_logits, model_targets

    @torch.no_grad()
    def validate_epoch(
        self,
        rec_model: BaseRecModel,
        model: MLPAttackerBaseModel,
        validation_dataloader: DataLoader,
        best_validation_scores: Dict,
        epoch: int,
        phase: str = "atk/val",
        verbose: bool = True,
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_epoch(
            rec_model, model, validation_dataloader, epoch, phase, training=False
        )
        results_dict = calculate_atk_metrics(
            self.eval_metrics, model_logits, model_targets, return_individual=False
        )
        best_validation_scores, stop_training = test_early_stopping_condition(
            model,
            epoch,
            self.early_stopping_criteria,
            results_dict,
            best_validation_scores,
            self.results_dir,
            verbose,
        )
        self.logger.log_value_dict(f"{phase}/metrics", results_dict, epoch)

        if self.use_wandb:
            wandb_write_to_run(self.wandb_runname, f"{phase}", best_validation_scores)
        return best_validation_scores, epoch_avg_loss, stop_training

    @torch.no_grad()
    def test_epoch(
        self,
        rec_model: BaseRecModel,
        model: MLPAttackerBaseModel,
        test_dataloader: DataLoader,
        phase="atk/test",
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_epoch(
            rec_model, model, test_dataloader, 1, phase, training=False
        )
        results_dict = calculate_atk_metrics(
            self.eval_metrics, model_logits, model_targets, return_individual=False
        )
        self.logger.log_value_dict(f"{phase}/metrics", results_dict, 1)
        self.logger.store()
        if self.use_wandb:
            wandb_write_to_run(self.wandb_runname, f"{phase}", results_dict)
        return results_dict

    def fit(
        self,
        rec_model: BaseRecModel,
        model: MLPAttackerBaseModel,
        dataloaders: Dict[str, Tuple[BaseDataset | DataLoader]],
        include_test: bool = False,
        is_verbose: bool = False,
    ):
        self._setup_optimizer(model)
        best_validation_scores = init_early_stopping_dict(self.early_stopping_criteria)
        train_dataloader = dataloaders["train"]
        valid_dataloader = dataloaders["val"]

        results_epoch_test = []
        for epoch in trange(
            self.n_epochs, desc="Epochs", position=1, leave=True, disable=not is_verbose
        ):

            self.run_epoch(
                rec_model=rec_model,
                model=model,
                dataloader=train_dataloader,
                epoch=epoch,
            )

            best_validation_scores, valid_avg_loss, stop_training = self.validate_epoch(
                rec_model,
                model,
                valid_dataloader,
                best_validation_scores,
                epoch,
                verbose=is_verbose,
            )

            if include_test:
                test_dataloader = dataloaders["test"]
                results_test_dict = self.test_epoch(rec_model, model, test_dataloader)
                results_epoch_test.append(results_test_dict)
            if epoch % self.store_model_every == 0 and epoch > 0:
                save_model(model, self.results_dir, f"model_epoch_{epoch}")

            if stop_training == True:
                print(f"Early stopping condition met epoch:{epoch}")
                save_model(model, self.results_dir, f"model_best_valid_{epoch}")
                break

            # Running scheduler on validation loss
            self.scheduler.step(valid_avg_loss)
            # Flush logger
            self.logger.store()

        yaml_dump(
            best_validation_scores,
            os.path.join(self.results_dir, "best_validation_scores.json"),
        )
        if self.store_last_model:
            save_model(model, self.results_dir, f"model_epoch_{epoch}")
        if include_test:
            yaml_dump(
                results_epoch_test,
                os.path.join(self.results_dir, "traning_test_scores.json"),
            )
        return best_validation_scores
