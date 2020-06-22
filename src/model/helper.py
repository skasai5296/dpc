from torch import optim

from model.criterion import BERTCPCLoss, ClassificationLoss, DPCLoss
from model.model import (BERTCPC, DPC, BERTCPCClassification,
                         DPCClassification, FineGrainedCPC,
                         FineGrainedCPCClassification)


def get_model_and_loss(CONFIG, finetune=False):
    assert CONFIG.model in ("DPC", "CPC", "FGCPC")
    if not finetune:
        if CONFIG.model == "DPC":
            model = DPC(
                CONFIG.input_size,
                CONFIG.hidden_size,
                CONFIG.kernel_size,
                CONFIG.num_layers,
                CONFIG.n_clip,
                CONFIG.pred_step,
                CONFIG.dropout,
            )
            criterion = DPCLoss()
        elif CONFIG.model == "CPC":
            model = BERTCPC(
                CONFIG.input_size,
                CONFIG.hidden_size,
                CONFIG.num_layers,
                CONFIG.num_heads,
                CONFIG.n_clip,
                CONFIG.dropout,
            )
            criterion = BERTCPCLoss(mse_weight=CONFIG.mse_weight)
        elif CONFIG.model == "FGCPC":
            model = FineGrainedCPC(
                CONFIG.input_size,
                CONFIG.hidden_size,
                7,
                CONFIG.num_layers,
                CONFIG.num_heads,
                CONFIG.n_clip,
                CONFIG.dropout,
            )
            criterion = BERTCPCLoss(mse_weight=CONFIG.mse_weight)
        else:
            raise NotImplementedError()
    else:
        if CONFIG.model == "DPC":
            model = DPCClassification(
                CONFIG.input_size,
                CONFIG.hidden_size,
                CONFIG.kernel_size,
                CONFIG.num_layers,
                CONFIG.n_clip,
                CONFIG.pred_step,
                CONFIG.dropout,
                700,
            )
        elif CONFIG.model == "CPC":
            model = BERTCPCClassification(
                CONFIG.input_size,
                CONFIG.hidden_size,
                CONFIG.num_layers,
                CONFIG.num_heads,
                CONFIG.n_clip,
                CONFIG.dropout,
                700,
            )
        elif CONFIG.model == "FGCPC":
            model = FineGrainedCPCClassification(
                CONFIG.input_size,
                CONFIG.hidden_size,
                7,
                CONFIG.num_layers,
                CONFIG.num_heads,
                CONFIG.n_clip,
                CONFIG.dropout,
                700,
            )
        else:
            raise NotImplementedError()
        criterion = ClassificationLoss()
    return model, criterion


def get_optimizer_and_scheduler(CONFIG, model):
    assert CONFIG.optimizer in ("adam", "sgd")
    assert CONFIG.scheduler in ("none", "step", "plateau")
    if CONFIG.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    elif CONFIG.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.lr,
            momentum=CONFIG.momentum,
            weight_decay=CONFIG.weight_decay,
        )
    else:
        raise NotImplementedError()
    if CONFIG.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=CONFIG.patience, gamma=CONFIG.dampening_rate,
        )
    elif CONFIG.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=CONFIG.dampening_rate,
            patience=CONFIG.patience,
            verbose=True,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.patience, gamma=1.0,)
    return optimizer, scheduler
