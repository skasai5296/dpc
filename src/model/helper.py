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
        criterion = ClassificationLoss()
    return model, criterion
