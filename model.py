import torch
import pytorch_lightning as pl
from torch.nn import Linear, SmoothL1Loss, ModuleList, ReLU, LeakyReLU, Identity, Sequential
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MultiAggregation


act_fn_dict = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'linear': Identity,
}


class GNNLayer(MessagePassing):

    def __init__(
            self,
            in_dim,
            out_dim,
            hidden_dim,
            aggr='sum',
            act_fn='relu',
            u_num_layers=2,
            **kwargs
    ):
        # combine multiple aggregations if needed
        if isinstance(aggr, str):
            aggr = [aggr]
        self.msg_expansion = len(aggr)
        aggr = MultiAggregation(aggr)
        super(GNNLayer, self).__init__(aggr=aggr, **kwargs)

        in_dims = [(1 + self.msg_expansion) * in_dim] + [hidden_dim] * (u_num_layers - 1)
        out_dims = [hidden_dim] * (u_num_layers - 1) + [out_dim]

        # construct update MLP
        act = act_fn_dict[act_fn]
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(Linear(in_dim, out_dim)),
            if i < u_num_layers - 1:
                layers.append(act())

        self.U = Sequential(*layers)

    def forward(self, x, edge_index, size=None):
        rec = self.propagate(edge_index, x=x, size=size)
        out = self.U(torch.cat([x, rec], dim=1))
        return out


class GNN(pl.LightningModule):

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, aggr, lr=0.001, weight_decay=1.0e-5, num_epochs=100, **kwargs):
        super(GNN, self).__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.out_dim = out_dim

        in_dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [out_dim]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(
                GNNLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=hidden_dim,
                    aggr=aggr,
                    **kwargs
                )
            )
        self.layers = ModuleList(layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.criterion = SmoothL1Loss(reduction='mean')

    def forward(self, data):
        y = data.x.view(-1, self.in_dim)
        for layer in self.layers:
            y = layer(y, edge_index=data.edge_index)
        return y

    def training_step(self, data, batch_idx, **kwargs):
        y_pred = self(data)

        y_pred = y_pred[data.mask].flatten()
        y_true = data.y[data.mask].flatten()
        loss = self.criterion(y_pred, y_true)
        mae = torch.abs(y_pred - y_true).mean()

        if self.global_rank == 0:
            self.log('Loss/Train', loss.cpu().detach(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('MAE/Train', mae.cpu().detach(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, data, batch_idx, **kwargs):
        y_pred = self(data)

        y_pred = y_pred[data.mask].flatten()
        y_true = data.y[data.mask].flatten()
        loss = self.criterion(y_pred, y_true)
        mae = torch.abs(y_pred - y_true).mean()
        batch_size = data.mask.sum()

        metrics = {'Loss/Valid': loss, 'MAE/Valid': mae}
        if self.global_rank == 0:
            for key, val in metrics.items():
                self.log(
                    key,
                    val.cpu().detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size
                )

        return metrics

    def predict_step(self, data, batch_idx, **kwargs):
        y_pred = self(data)
        y_pred = y_pred[data.mask].flatten()
        y_true = data.y[data.mask].flatten()
        abs_err = torch.abs(y_pred - y_true)
        return abs_err

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_epochs, verbose=True)
        return {"optimizer": opt, "lr_scheduler": sched}
