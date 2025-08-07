from src.trainer.BaseTrainer import BaseTrainer
from src.data.BaseDataset import BaseDataset
from src.utils.helper import yaml_dump, save_model
from src.recsys_models.BaseRecModel import (
    BaseRecModel,
    AdvBaseRecModel,
    PairWiseRecModel,
)
from torch.utils.data import DataLoader
import torch
from src.evaluation.recommendation import calculate_recommendation_metrics
import wandb
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
import os
from tqdm import trange, tqdm


class AETrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def fit(
        self,
        model: BaseRecModel,
        dataloaders: Dict[str, DataLoader],
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

            self.run_epoch(model=model, dataloader=train_dataloader, epoch=epoch)

            best_validation_scores, valid_avg_loss, stop_training = self.validate_epoch(
                model,
                valid_dataloader,
                best_validation_scores,
                epoch,
            )

            if include_test:
                test_dataloader = dataloaders["test"]
                results_test_dict = self.test_epoch(model, test_dataloader)
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

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for indices, model_input, targets, _ in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(indices)
            sample_count += n_samples
            targets = targets.to(device)
            model_input = model_input.to(device)

            logits = model(model_input)

            loss, loss_dict = model.calc_loss(logits, targets)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not training:
                # Masking logits for evaluation
                masked_logits = logits[0]
                masked_logits[model_input.nonzero(as_tuple=True)] = 0.0
                model_logits.append(masked_logits.detach())
                model_targets.append(targets.detach())

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
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets

    @torch.no_grad()
    def validate_epoch(
        self,
        model: BaseRecModel,
        validation_dataloader: DataLoader,
        best_validation_scores: Dict,
        epoch: int,
        phase: str = "val",
        verbose: bool = True,
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_epoch(
            model, validation_dataloader, epoch, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            [self.eval_metrics[0]],
            model_logits,
            model_targets,
            [self.metrics_top_k[0]],
            flatten_results=True,
            return_individual=False,
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
        self.logger.log_value_dict(f"val/metrics", results_dict, epoch)
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in results_dict.items()})
        return best_validation_scores, epoch_avg_loss, stop_training

    @torch.no_grad()
    def test_epoch(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        phase="test",
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_epoch(
            model, test_dataloader, 1, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            self.eval_metrics,
            model_logits,
            model_targets,
            self.metrics_top_k,
            flatten_results=True,
            return_individual=False,
        )
        self.logger.log_value_dict(f"test/metrics", results_dict, 1)
        self.logger.store()
        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in results_dict.items()})
        return results_dict


class UserFeatureAETrainer(AETrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):
        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for indices, model_input, targets, user_features in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(indices)
            sample_count += n_samples
            targets = targets.to(device)
            model_input = model_input.to(device)
            user_features = [uf.to(device) for uf in user_features]

            logits = model(model_input)

            loss, loss_dict = model.calc_loss(logits, targets, user_features)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not training:
                # Masking logits for evaluation
                masked_logits = logits[0]
                masked_logits[model_input.nonzero(as_tuple=True)] = 0.0
                model_logits.append(masked_logits.detach().cpu())
                model_targets.append(targets.detach().cpu())

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
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        # print([f"{phase}", epoch_avg_loss])
        return epoch_avg_loss, model_logits, model_targets


class AdvAETrainer(AETrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.adv_optimizer_config = self.trainer_config.get("adv_optimizer_config")

    def _setup_optimizer(self, model: torch.nn.Module):
        self.optimizer = self.optimizer_class(
            [
                {
                    "params": model.encoder.parameters(),
                    **self.trainer_config["optimizer_config"],
                },
                {
                    "params": model.decoder.parameters(),
                    **self.trainer_config["optimizer_config"],
                },
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.trainer_config["scheduler"]
        )

        self.adv_optimizer = self.optimizer_class(
            model.adversaries.parameters(),
            **self.trainer_config["adv_optimizer_config"],
        )
        self.adv_scheduler = None
        if adv_sched_config := self.trainer_config.get("adv_scheduler"):
            self.adv_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.adv_optimizer, **adv_sched_config
            )

    def fit(
        self,
        model: AdvBaseRecModel,
        dataloaders: Dict[str, DataLoader],
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

            self.run_epoch(model=model, dataloader=train_dataloader, epoch=epoch)

            (
                best_validation_scores,
                valid_avg_loss,
                valid_avg_adv_loss,
                stop_training,
            ) = self.validate_epoch(
                model,
                valid_dataloader,
                best_validation_scores,
                epoch,
            )

            if include_test:
                test_dataloader = dataloaders["test"]
                results_test_dict = self.test_epoch(model, test_dataloader)
                results_epoch_test.append(results_test_dict)
            if epoch % self.store_model_every == 0 and epoch > 0:
                save_model(model, self.results_dir, f"model_epoch_{epoch}")

            if stop_training == True:
                print(f"Early stopping condition met epoch:{epoch}")
                save_model(model, self.results_dir, f"model_best_valid_{epoch}")
                break

            # Running scheduler on validation loss
            self.scheduler.step(valid_avg_loss)
            if model.adversaries_enabled:
                if self.adv_scheduler != None:
                    self.adv_scheduler.step(valid_avg_adv_loss)
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

    def run_epoch(
        self,
        model: AdvBaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):
        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss, adv_train_loss = 0, 0.0, 0.0
        for indices, model_input, targets, user_features in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(indices)
            sample_count += n_samples
            targets = targets.to(device)
            model_input = model_input.to(device)
            user_features = [uf.to(device) for uf in user_features]

            logits = model(model_input)

            loss, adv_loss, loss_dict = model.calc_loss(logits, targets, user_features)

            train_loss += loss * n_samples
            adv_train_loss += adv_loss * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            if training:
                self.optimizer.zero_grad()
                if model.adversaries_enabled:
                    self.adv_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if model.adversaries_enabled:
                    self.adv_optimizer.step()

            if not training:
                # Masking logits for evaluation
                masked_logits = logits[0]
                masked_logits[model_input.nonzero(as_tuple=True)] = 0.0
                model_logits.append(masked_logits.detach().cpu())
                model_targets.append(targets.detach().cpu())

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
        adv_epoch_avg_loss = adv_train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        # print([f"{phase}", epoch_avg_loss])
        return epoch_avg_loss, adv_epoch_avg_loss, model_logits, model_targets

    @torch.no_grad()
    def validate_epoch(
        self,
        model: AdvBaseRecModel,
        validation_dataloader: DataLoader,
        best_validation_scores: Dict,
        epoch: int,
        phase: str = "val",
        verbose: bool = True,
    ):
        epoch_avg_loss, adv_epoch_avg_loss, model_logits, model_targets = (
            self.run_epoch(model, validation_dataloader, epoch, phase, training=False)
        )
        results_dict = calculate_recommendation_metrics(
            [self.eval_metrics[0]],
            model_logits,
            model_targets,
            [self.metrics_top_k[0]],
            flatten_results=True,
            return_individual=False,
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
        self.logger.log_value_dict(f"val/metrics", results_dict, epoch)
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in results_dict.items()})
        return best_validation_scores, epoch_avg_loss, adv_epoch_avg_loss, stop_training

    @torch.no_grad()
    def test_epoch(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        phase="test",
    ):
        epoch_avg_loss, adv_epoch_avg_loss, model_logits, model_targets = (
            self.run_epoch(model, test_dataloader, 1, phase, training=False)
        )
        results_dict = calculate_recommendation_metrics(
            self.eval_metrics,
            model_logits,
            model_targets,
            self.metrics_top_k,
            flatten_results=True,
            return_individual=False,
        )
        self.logger.log_value_dict(f"test/metrics", results_dict, 1)
        self.logger.store()
        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in results_dict.items()})
        return results_dict


class PointwiseTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def fit(
        self,
        model: BaseRecModel,
        dataloaders: Dict[str, DataLoader],
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

            self.run_epoch(model=model, dataloader=train_dataloader, epoch=epoch)

            best_validation_scores, valid_avg_loss, stop_training = self.validate_epoch(
                model,
                valid_dataloader,
                best_validation_scores,
                epoch,
            )

            if include_test:
                test_dataloader = dataloaders["test"]
                results_test_dict = self.test_epoch(model, test_dataloader)
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

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            neg_item_ids,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            # logits = model(user_ids, pos_item_ids, neg_item_ids)

            loss, loss_dict = model.calc_loss(user_ids, pos_item_ids)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets

    def run_eval_epoch(
        self,
        model: PairWiseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            _,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)

            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            logits = model.full_predict(user_ids)
            loss = 0.0
            loss_dict = {"loss": 0.0}

            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )
            # print(logits.size(), exclude_items.size())
            masked_logits = logits
            masked_logits[exclude_items] = -torch.inf
            model_logits.append(masked_logits.detach())
            model_targets.append(pos_item_ids.detach())

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        model_logits = torch.cat(model_logits)
        model_targets = torch.cat(model_targets)
        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets

    @torch.no_grad()
    def validate_epoch(
        self,
        model: BaseRecModel,
        validation_dataloader: DataLoader,
        best_validation_scores: Dict,
        epoch: int,
        phase: str = "val",
        verbose: bool = True,
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_eval_epoch(
            model, validation_dataloader, epoch, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            [self.eval_metrics[0]],
            model_logits,
            model_targets,
            [self.metrics_top_k[0]],
            flatten_results=True,
            return_individual=False,
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
        self.logger.log_value_dict(f"val/metrics", results_dict, epoch)
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in results_dict.items()})
        return best_validation_scores, epoch_avg_loss, stop_training

    @torch.no_grad()
    def test_epoch(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        phase="test",
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_eval_epoch(
            model, test_dataloader, 1, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            self.eval_metrics,
            model_logits,
            model_targets,
            self.metrics_top_k,
            flatten_results=True,
            return_individual=False,
        )
        self.logger.log_value_dict(f"test/metrics", results_dict, 1)
        self.logger.store()
        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in results_dict.items()})
        return results_dict


class FeauturePointwiseTrainer(PointwiseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            neg_item_ids,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)
            user_features = [uf.to(device) for uf in user_features]
            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            # logits = model(user_ids, pos_item_ids, neg_item_ids)

            loss, loss_dict = model.calc_loss(user_ids, pos_item_ids, user_features)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets


class PairwiseTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def fit(
        self,
        model: BaseRecModel,
        dataloaders: Dict[str, DataLoader],
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

            self.run_epoch(model=model, dataloader=train_dataloader, epoch=epoch)

            best_validation_scores, valid_avg_loss, stop_training = self.validate_epoch(
                model,
                valid_dataloader,
                best_validation_scores,
                epoch,
            )

            if include_test:
                test_dataloader = dataloaders["test"]
                results_test_dict = self.test_epoch(model, test_dataloader)
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

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            neg_item_ids,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)
            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            # logits = model(user_ids, pos_item_ids, neg_item_ids)

            loss, loss_dict = model.calc_loss(user_ids, pos_item_ids, neg_item_ids)

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets

    def run_eval_epoch(
        self,
        model: PairWiseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            _,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)

            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            logits = model.full_predict(user_ids)
            loss = 0.0
            loss_dict = {"loss": 0.0}

            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )
            # print(logits.size(), exclude_items.size())
            masked_logits = logits
            masked_logits[exclude_items] = -torch.inf
            model_logits.append(masked_logits.detach())
            model_targets.append(pos_item_ids.detach())

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        model_logits = torch.cat(model_logits)
        model_targets = torch.cat(model_targets)
        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets

    @torch.no_grad()
    def validate_epoch(
        self,
        model: BaseRecModel,
        validation_dataloader: DataLoader,
        best_validation_scores: Dict,
        epoch: int,
        phase: str = "val",
        verbose: bool = True,
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_eval_epoch(
            model, validation_dataloader, epoch, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            [self.eval_metrics[0]],
            model_logits,
            model_targets,
            [self.metrics_top_k[0]],
            flatten_results=True,
            return_individual=False,
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
        self.logger.log_value_dict(f"val/metrics", results_dict, epoch)
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in results_dict.items()})
        return best_validation_scores, epoch_avg_loss, stop_training

    @torch.no_grad()
    def test_epoch(
        self,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        phase="test",
    ):
        epoch_avg_loss, model_logits, model_targets = self.run_eval_epoch(
            model, test_dataloader, 1, phase, training=False
        )
        results_dict = calculate_recommendation_metrics(
            self.eval_metrics,
            model_logits,
            model_targets,
            self.metrics_top_k,
            flatten_results=True,
            return_individual=False,
        )
        self.logger.log_value_dict(f"test/metrics", results_dict, 1)
        self.logger.store()
        if self.use_wandb:
            wandb.log({f"test/{k}": v for k, v in results_dict.items()})
        return results_dict


class FeauturePairwiseTrainer(PairwiseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def run_epoch(
        self,
        model: BaseRecModel,
        dataloader: DataLoader,
        epoch: int,
        phase: str = "train",
        training: bool = True,
    ):

        if training:
            model.train()
        else:
            model.eval()

        batch_count = len(dataloader) * epoch
        train_loss_dict = defaultdict(lambda: 0)
        model_logits, model_targets = [], []
        # initialize variables for result accumulation
        sample_count, train_loss = 0, 0.0
        for (
            user_ids,
            pos_item_ids,
            neg_item_ids,
            exclude_items,
            (user_features, item_features),
        ) in tqdm(
            dataloader, desc="Training steps", position=2, leave=True, disable=True
        ):
            device = self.device

            n_samples = len(user_ids)
            sample_count += n_samples
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)
            user_features = [uf.to(device) for uf in user_features]
            # print(user_ids.size(), pos_item_ids.size(), neg_item_ids.size())
            # logits = model(user_ids, pos_item_ids, neg_item_ids)

            loss, loss_dict = model.calc_loss(
                user_ids, pos_item_ids, neg_item_ids, user_features
            )

            train_loss += loss.detach() * n_samples
            train_loss_dict = add_to_dict(
                train_loss_dict, loss_dict, multiplier=n_samples
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log results inside batch to get a better idea whether the model works
            if batch_count % self.log_every_n_batches == 0:
                self.logger.log_value(f"{phase}/batch_loss", loss, batch_count)
                self.logger.log_value_dict(
                    f"{phase}/batch_loss", loss_dict, batch_count
                )
            batch_count += 1

        epoch_avg_loss = train_loss / sample_count

        self.logger.log_value(f"{phase}/loss", epoch_avg_loss, epoch)
        self.logger.log_value_dict(
            f"{phase}/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch
        )
        if self.use_wandb:
            wandb.log({"train/loss": epoch_avg_loss, "epoch": epoch})
        return epoch_avg_loss, model_logits, model_targets
