import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl

class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)


    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = nn.CrossEntropyLoss()(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels, true_labels, loss

    def training_step(self, batch, _):
        predicted_labels, true_labels, loss = self._classifier_step(batch)
        self.train_accuracy(predicted_labels, true_labels)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, _):
        predicted_labels, true_labels, loss = self._classifier_step(batch)
        self.val_accuracy(predicted_labels, true_labels)
        self.log('val_loss', loss)
        self.log('val_accuracy', self.val_accuracy, on_step=True, on_epoch=False, prog_bar=True)

    def test_step(self, batch, _):
        predicted_labels, true_labels, loss = self._classifier_step(batch)
        self.test_accuracy(predicted_labels, true_labels)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=False)


    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer