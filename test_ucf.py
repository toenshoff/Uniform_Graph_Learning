from argparse import ArgumentParser
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torch_geometric.loader import DataLoader
from model import GNN
from train_ucf import generate_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum k")
    parser.add_argument("--max_k", type=int, default=1000, help="Maximum k")
    parser.add_argument("--step_k", type=int, default=1, help="Step size of k")
    parser.add_argument("--min_c", type=int, default=1, help="Minimum c")
    parser.add_argument("--max_c", type=int, default=1000, help="Maximum c")
    parser.add_argument("--step_c", type=int, default=1, help="Step size of c")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers in the data loaders")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    k_iter = list(range(args.min_k, args.max_k + args.step_k, args.step_k))
    c_iter = list(range(args.min_c, args.max_c + args.step_c, args.step_c))
    data_list = generate_data(k_iter, c_iter)

    test_loader = DataLoader(
        dataset=data_list,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    model = GNN.load_from_checkpoint(args.ckpt_path)
    model.to(device)

    with torch.inference_mode():

        abs_err_list = []
        for data in tqdm(test_loader):
            data.to(device)
            abs_err = model.predict_step(data, None)
            abs_err = abs_err.cpu()
            abs_err_list.append(abs_err)

        abs_err = torch.cat(abs_err_list, dim=0)
        abs_err = abs_err.view(len(k_iter), len(c_iter)).numpy()

    df = pd.DataFrame(abs_err, columns=c_iter, index=k_iter)

    csv_path = os.path.join(os.path.dirname(args.ckpt_path), 'test.csv')
    df.to_csv(csv_path)
