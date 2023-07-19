from torch.nn import functional as F
from transformers import Trainer


def distillation_loss(predictions, targets, temperature):
    p = F.log_softmax(predictions / temperature, dim=1)
    return F.kl_div(p, targets, reduction='batchmean') * (temperature ** 2) / predictions.shape[0]


class DistillTrainer(Trainer):
    def __init__(self,model, args, temperature=2.0):
        super().__init__(model = model, args=args)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = distillation_loss(logits, labels, temperature=self.temperature)
        return (loss, outputs) if return_outputs else loss


class FineTuningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        return (loss, outputs) if return_outputs else loss
