import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from ..pipeline.GateComparisionLogger import GateComparisionLogger
from ..pipeline.ValidationService import ValidationService
from ..datastructures.configs.dataproviderconfig import DataProviderConfig
from ..loader.IO.model_file_handler import ModelFileHandler

from ..pipeline.DataProvider import DataProvider
from ..pipeline.DeviceManager import DeviceManager
from ..utils.reproducibility_helper import setSeeds


from ..loss.flowsample_gating_loss import flowsample_gating_loss
from ..loss.auxloss_helper import compute_auxiliary_loss, register_perciever_forward_hock
from .ModelFactory import ModelFactory
from .WandbLogger import WandbLogger
from ..loader.DataSet.FileTypeFolderDataSet import FileTypeFolderDataSet
from ..datastructures.configs.trainconfig import TrainConfig
import numpy as np
from torchsummary import summary


class Trainer:
    """
    Main class for model training.
    Executes training based on the given TrainConfig.
    Loads data, handles loss computation, logging and validation.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.prepare_training()

    def prepare_training(self):

        # load dataset
        self.train_dataset = FileTypeFolderDataSet.init_from_config(self.config.train_data)
        self.validate_dataset = FileTypeFolderDataSet.init_from_config(self.config.validation_data)
        # load model from storage / init model
        modelFactory = ModelFactory(self.config.model_factory)
        self.model = modelFactory.create_instance()
        self.model_file_handler = ModelFileHandler(self.config.model_storage)

        if self.config.model_storage.load_stats_from_file:
            self.model_file_handler.load_state_from_file(self.model, True)

        print(self.model)

        all_params = set(self.model.parameters())
        wd_params = set()
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                wd_params.add(m.weight)

        no_wd = all_params - wd_params
        self.optimizer = torch.optim.AdamW([{'params': list(no_wd), 'weight_decay': 0}, {'params': list(wd_params), 'weight_decay': self.config.train_params.weight_decay}], lr=self.config.train_params.learning_rate, weight_decay=self.config.train_params.weight_decay)
        self.logger = WandbLogger(self.config.wandb_config, {** self.config.model_factory.params, **self.config.train_params.__dict__, **self.config.default_retrieve_options.__dict__})
        self.logger.set_config_value("n_samples_train", len(self.train_dataset.fileIndexes))
        self.device_manager = DeviceManager(self.config.gpu_name)
        self.plotlogger = GateComparisionLogger(self.logger, self.config.default_retrieve_options)
        self.config.default_retrieve_options.preprocess_data()
        self.validationSrv = ValidationService(self.config.default_retrieve_options, self.config.gpu_name)


        setSeeds(self.config.train_params.random_seed)

    def train(self):

        loss_fn = flowsample_gating_loss(polygon_weight=self.config.train_params.polygon_loss_weight)

        loss_result_list = []

        dataprovider = DataProvider(self.train_dataset, self.device_manager, DataProviderConfig(n_samples_to_load=self.config.train_params.training_batchsize, batchsize=self.config.train_params.training_batchsize), self.config.default_retrieve_options)
        dataloader = DataLoader(dataprovider, num_workers=self.config.n_workers, batch_size=self.config.train_params.training_batchsize, collate_fn=DataProvider.collate_fn)
        summary(self.model, (self.config.default_retrieve_options.events_seq_length, 10), device="cpu")

        self.model.train(True)
        self.device_manager.move_to_gpu(self.model)
        self.logger.watch_model(self.model)
        # torch.autograd.set_detect_anomaly(True)

        if self.config.train_params.use_auxiliary_loss:
            intermed_res = []
            register_perciever_forward_hock(intermed_res, self.model)

        for i in range(1, self.config.train_params.n_training_epochs + 1):
            print(f"\ntraining in epoch '{i}'\n augmentation_disabled: {self.config.default_retrieve_options.augmentation_config.disable_augmentation}")

            epoch_losses_list = []
            aux_loss_list = []
            polygon_loss_list = []
            loss = None

            self.logger.log({"epoch": i}, commit=False)
            polygons_pred = 0
            for flowsample_tensors in dataloader:
                self.optimizer.zero_grad()
                if self.config.train_params.use_auxiliary_loss:
                    intermed_res.clear()

                try:
                    flowsample_tensors = dataprovider.move_sample_to_gpu(flowsample_tensors)
                    polygons_pred = self.model(flowsample_tensors.events, flowsample_tensors.padding_mask)

                    if self.config.train_params.use_batch_loss == False and len(flowsample_tensors.polygons.shape) == 4:
                        for in_batch_idx in range(0, flowsample_tensors.events.shape[0]):
                            # the loss is not computed batchwise --> loop
                            loss = loss_fn(polygons_pred=polygons_pred[in_batch_idx], polygons_gt=flowsample_tensors.polygons[in_batch_idx])

                            summed_polygon_loss = torch.sum(loss.point_loss)
                            overall_loss = summed_polygon_loss

                            if self.config.train_params.use_auxiliary_loss:
                                aux_loss = compute_auxiliary_loss(intermed_res, flowsample_tensors.polygons, in_batch_idx, loss_fn, self.config.train_params.aux_loss_increasing_weight)
                                overall_loss = (1- self.config.train_params.auxiliary_loss_weight) * overall_loss + self.config.train_params.auxiliary_loss_weight * aux_loss
                                aux_loss_list.append(aux_loss.detach().cpu().numpy())

                            loss_result_list.append(overall_loss)
                            epoch_losses_list.append(overall_loss.detach().cpu().numpy())
                            polygon_loss_list.append(summed_polygon_loss.detach().cpu().numpy())

                        summed_loss = torch.Tensor.sum(torch.stack(loss_result_list))
                    else:
                        # print(polygons_pred.shape)
                        # print(flowsample_tensors.polygons.shape)
                        summed_loss = loss_fn(polygons_pred, flowsample_tensors.polygons).point_loss
                        epoch_losses_list.append(summed_loss.detach().cpu().numpy())

                    summed_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train_params.clip_norm)

                    self.optimizer.step()
                    loss_result_list = []
                    summed_loss = None
                    if self.config.train_params.use_auxiliary_loss:
                        intermed_res.clear()

                except Exception as ex:
                    print(ex)
                    loss_result_list = []
                    summed_loss = None
                    if self.config.train_params.use_auxiliary_loss:
                        intermed_res.clear()
            # validation
            if i % self.config.train_params.validation_interval == 1 or i == self.config.train_params.n_training_epochs:

                polygons_pred = self.device_manager.move_to_cpu(polygons_pred[0])
                if loss is not None:
                    loss.point_loss = self.device_manager.move_to_cpu(loss.point_loss)
                if len(flowsample_tensors.sample_names) > 1:
                    gt_polygons = self.device_manager.move_to_cpu(flowsample_tensors.polygons[0])
                else:
                    gt_polygons = self.device_manager.move_to_cpu(flowsample_tensors.polygons)

                self.plotlogger.log_all_gate_comparision_figures(flowsample_tensors.sample_names[0], polygons_pred, gt_polygons, loss)

                self.model.train(False)

                print("\n\n validating model on validation dataset ... \n\n")
                res_per_gate = self.validationSrv.validate_dataset(self.model, self.validate_dataset)
                colnames = list(res_per_gate.columns.array)
                self.logger.log_table(key="validation_result", columns=colnames, data=[list(el) for el in res_per_gate.values], commit=False)
                mean_val_res = list(res_per_gate.iloc[:, 1:].mean(axis=0))
                self.logger.log({"avg_validation_result": mean_val_res}, commit=False)
                for log_idx, metric_name in enumerate(colnames[1:-3]):
                    self.logger.log({f"avg_validation_{metric_name}_f1": mean_val_res[log_idx]}, commit=False)

                print(f"successfully validated model on validation dataset using {len(res_per_gate)} samples\n")

                if self.config.train_params.run_additional_validation_on_train_data:
                    print("\n\n validating model on train dataset ... \n\n")
                    res_per_gate = self.validationSrv.validate_dataset(self.model, self.train_dataset)
                    colnames = list(res_per_gate.columns.array)
                    self.logger.log_table(key="train_val_result", columns=colnames, data=[list(el) for el in res_per_gate.values], commit=False)
                    mean_val_res = list(res_per_gate.iloc[:, 1:].mean(axis=0))
                    self.logger.log({"avg_train_validation_result": mean_val_res}, commit=False)
                    for log_idx, metric_name in enumerate(colnames[1:-3]):
                        self.logger.log({f"avg_train_validation_{metric_name}_f1": mean_val_res[log_idx]}, commit=False)

                    print(f"successfully validated model on train dataset using {len(res_per_gate)} samples\n")

                self.model.train(True)

            # saving
            if i % self.config.train_params.saving_interval == 1:
                self.model_file_handler = ModelFileHandler(self.config.model_storage)
                self.model_file_handler.save_model_state_to_file(self.model)

            epoch_loss_mean = np.mean(np.array(epoch_losses_list))
            polygon_loss_mean = np.mean(np.array(polygon_loss_list))

            losses_dict = {"overall_polygon_loss": polygon_loss_mean, "overall_epoch_loss": epoch_loss_mean}

            if len(aux_loss_list) > 0:
                aux_loss_mean = np.mean(np.array(aux_loss_list))
                losses_dict["overall_auxiliary_loss"] = aux_loss_mean

            self.logger.log(losses_dict)
            epoch_losses_list = []

        print("finished training!")
        self.logger.finish()
