from dataclasses import dataclass


@dataclass
class TrainParams:
    learning_rate: float
    weight_decay: float
    validation_interval: int
    saving_interval : int
    n_training_epochs: int
    clip_norm : float
    polygon_loss_weight : float
    training_batchsize: int = 1
    run_additional_validation_on_train_data : bool = True
    use_gpu: bool = True
    random_seed : int = 42
    use_batch_loss : bool = False
    use_auxiliary_loss : bool = False
    auxiliary_loss_weight : float = 0.3
    aux_loss_increasing_weight : bool = False

