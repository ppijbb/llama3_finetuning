import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

from .student import StudentModel
from .teacher import TeacherModel
from .data_module import DistillationDataModule


class DistillationModel(pl.LightningModule):
    def __init__(self, student_model, teacher_model, learning_rate=1e-3):
        super(DistillationModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        data, _ = batch
        student_output = self.student_model(data)
        with torch.no_grad():
            teacher_output = self.teacher_model(data)
        loss = self.loss_fn(
            student_output.log_softmax(dim=-1), 
            teacher_output.softmax(dim=-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.student_model.parameters(), lr=self.learning_rate)

def train_distillation(batch_size=32, max_epochs=10):
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        batch_size=batch_size,
        strategy='deepspeed',
        precision='bf16',
        accumulate_grad_batches=4)
    trainer.fit(
        model=DistillationModel(
            student_model=StudentModel(input_dim=784, hidden_dim=16, output_dim=10), 
            teacher_model=TeacherModel(input_dim=784, hidden_dim=128, output_dim=10)),
        datamodule=DistillationDataModule(
            train_data="",
            val_data="",
            test_data="",
            batch_size=batch_size))
