# libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score
import lightning as L

class DSArray(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_classes = len(list(set(list(Y))))
        self.inp_ft = X.shape[1]

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_i = torch.from_numpy(np.array(self.X[idx])).float()
        y_i = torch.from_numpy(np.array(self.Y[idx])).long()

        return X_i, y_i


class ResModule(nn.Module):
    def __init__(self, inp_ft, out_ft, dropout_val):
        super(ResModule, self).__init__()

        self.fc1 = nn.Linear(inp_ft, out_ft)
        self.fc2 = nn.Linear(out_ft, out_ft)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_val)

        if inp_ft != out_ft:
            self.res_connection = nn.Linear(inp_ft, out_ft)
        else:
            self.res_connection = None

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.res_connection is not None:
            identity = self.res_connection(identity)

        out += identity
        out = self.relu(out)

        return out

class MLP(L.LightningModule):
    def __init__(self, inp_ft, num_classes, dropout_val, num_epochs, bs, lr, num_layers, num_nodes):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i == 0:
                self.mlp.add_module(f"res_{i}", ResModule(inp_ft, num_nodes, dropout_val))
            else:
                self.mlp.add_module(f"res_{i}", ResModule(num_nodes, num_nodes, dropout_val))
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(dropout_val))

        self.mlp.add_module("output", nn.Linear(num_nodes, num_classes))

        self.loss = nn.CrossEntropyLoss()

        self.lr = lr
        self.bs = bs
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_epochs = num_epochs
        self.y_test_pred = []
        self.y_test_true = []

    def forward(self, x):
        out = self.mlp(x)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y = batch
        # pass through model
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        return loss, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        monitor = 'val/loss'
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": monitor}}

    def training_step(self, batch):

        loss, _ = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, _ = self._get_loss(batch)

        self.log('val/loss', loss)

        return loss
    
    def test_step(self, batch):
        x, y = batch 
        for y_i in y.cpu().detach().numpy():
            self.y_test_true.append(y_i)
        loss, yp = self._get_loss(batch)
        for yp_i in yp.cpu().detach().numpy():
            self.y_test_pred.append(np.argmax(yp_i))

        self.log('test/loss', loss)

        return loss
    
    def on_test_epoch_end(self):
        f1_score_task = f1_score(self.y_test_true, self.y_test_pred, average='macro')
        balacc_task = balanced_accuracy_score(self.y_test_true, self.y_test_pred)

        print(f"F1 Score: {f1_score_task}, Balanced Accuracy: {balacc_task}")

        self.log('test/f1', f1_score_task)
        self.log('test/balacc', balacc_task)

def trainMLP(inp_ft, num_classes, num_epochs, bs, lr, save_loc, train_loader, test_loader, val_loader, dropout_val, num_layers, num_nodes):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=1,
        max_epochs=num_epochs,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val/loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val/loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    model = MLP(inp_ft, num_classes, dropout_val, num_epochs, bs, lr, num_layers, num_nodes)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total MLP parameters: ", pytorch_total_params)

    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    result = trainer.test(model, dataloaders=test_loader, verbose=False)

    return model, result