import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

class DistillationDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

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
        loss = self.loss_fn(student_output.log_softmax(dim=-1), teacher_output.softmax(dim=-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.student_model.parameters(), lr=self.learning_rate)

def train_distillation(student_model, teacher_model, train_data, train_targets, batch_size=32, max_epochs=10):
    dataset = DistillationDataset(train_data, train_targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DistillationModel(student_model, teacher_model)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader)

# Example usage:
# student_model = YourStudentModel()
# teacher_model = YourTeacherModel()
# train_data = YourTrainingData()
# train_targets = YourTrainingTargets()
# train_distillation(student_model, teacher_model, train_data, train_targets)