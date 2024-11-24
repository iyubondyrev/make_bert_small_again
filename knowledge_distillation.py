import torch
import torch.nn as nn
from transformers import Trainer
from torch.utils.data import DataLoader


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        teacher_probs = torch.log_softmax(teacher_logits / self.temperature, dim=-1)
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(input=student_probs, target=teacher_probs) * (self.temperature ** 2)

        hard_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True
        )

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        
        student_logits = model(**inputs).logits
        
        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits
        
        loss = self.loss_fn(student_logits, teacher_logits, labels)
        return (loss, student_logits) if return_outputs else loss