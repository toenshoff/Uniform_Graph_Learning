from argparse import ArgumentParser
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader
from model import GNN, act_fn_dict
from data import ucf_data


def generate_data(k_iter, c_iter):
    data_list = []
    for k in k_iter:
        for c in c_iter:
            data = ucf_data(k, c)
            data_list.append(data)
    return data_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("--aggr", type=str, default=['mean'], nargs='+', choices=['mean', 'max', 'sum'], help="Aggregation fn")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden Dimension")
    parser.add_argument("--act_fn", type=str, default='relu', choices=list(act_fn_dict.keys()), help="Activation")
    parser.add_argument("--u_num_layers", type=int, default=2, help="Number of layer in the update function")
    parser.add_argument("--lr", type=float, default=1.e-4, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1.e-5, help="Weight Decay")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum k")
    parser.add_argument("--max_k", type=int, default=100, help="Maximum k")
    parser.add_argument("--step_k", type=int, default=1, help="Step size of k")
    parser.add_argument("--min_c", type=int, default=1, help="Minimum c")
    parser.add_argument("--max_c", type=int, default=100, help="Maximum c")
    parser.add_argument("--step_c", type=int, default=1, help="Step size of c")
    parser.add_argument("--num_valid_data", type=int, default=500, help="Number of validation instances")
    parser.add_argument("--train_batch_size", type=int, default=100, help="Training Batch Size")
    parser.add_argument("--valid_batch_size", type=int, default=100, help="Validation Batch Size")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in the data loaders")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    kwargs = vars(args)

    k_iter = list(range(args.min_k, args.max_k + args.step_k, args.step_k))
    c_iter = list(range(args.min_c, args.max_c + args.step_c, args.step_c))
    data_list = generate_data(k_iter, c_iter)
    np.random.shuffle(data_list)

    train_loader = DataLoader(
        dataset=data_list[args.num_valid_data:],
        num_workers=args.num_workers,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        dataset=data_list[:args.num_valid_data],
        num_workers=args.num_workers,
        batch_size=args.valid_batch_size
    )

    model = GNN(in_dim=1, out_dim=1, **kwargs)

    logger = TensorBoardLogger(args.model_dir, name='', default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_dir, save_top_k=1, monitor="MAE/Valid", filename='best')
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=args.num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=10,
        precision=32,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, valid_loader)
