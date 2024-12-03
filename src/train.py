import torch
import argparse

import numpy as np
from tqdm import tqdm
import random

from model import SASRec, GRU4Rec

from util import *

import click

@click.command()
@click.option('--base', type=str, default='SASRec')
@click.option('--data', type=str, default='ml-1m')
@click.option('--epoch', type=int, default=200)
@click.option('--alpha', type=float, default=0.0)
def train_and_save(base, data, epoch, alpha):

    # set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read dataset
    train_UI, train_UT, user_num, item_num, _, _, item_prob, interval, cnt = data_split_userseq(f'../data/{data}.pkl')

    # set hyperparameters for a base model
    if base == 'SASRec' or base == 'sas':
        base = 'SASRec'
        args = argparse.Namespace(
            batch_size=128,
            lr=0.001,
            maxlen=200,
            hidden_units=50,
            num_blocks=2,
            num_epochs=epoch+1,
            num_heads=1,
            dropout_rate=0.5,
            l2_emb=0.0,
            device=device,
            inference_only=False,
            state_dict_path=None
        )
        model = SASRec(user_num, item_num, args).to(device)
    elif base == 'GRU4Rec' or base == 'gru':
        base == 'GRU4Rec'
        args = argparse.Namespace(
            batch_size=128,
            lr=0.001,
            maxlen=200,
            num_epochs=epoch+1,
            hidden_units=50,
            l2_emb=0.0,
            device=device
        )
        model = GRU4Rec(
            device=args.device,
            hidden_size = args.hidden_units,
            embedding_dim = args.hidden_units,
            item_num=item_num,
            state_size=0,
            action_dim = args.hidden_units,
            gru_layers=1,
            use_packed_seq=False,
            train_pad_embed=True,
            padding_idx=0,
        ).to(device)
        
    # initialize parameters
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.train()

    # initialize an optimizer
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # path to save a trained model
    if base == 'SASRec':
        name = f'../model/{data}/sas_alpha={alpha}_epoch={epoch}'
    elif base == 'GRU4Rec':
        name = f'../model/{data}/gru_alpha={alpha}_epoch={epoch}'

    # initialize mini-batch samplers
    sampler = WarpSampler(train_UI, train_UT, user_num, item_num, item_prob, interval, cnt, alpha=alpha, 
                        maxlen=args.maxlen, batch_size=args.batch_size, n_workers=16)
    num_batch = user_num // args.batch_size

    # train a model
    print('start training')
    for epoch in tqdm(range(args.num_epochs+1)):
        model.train()
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            if base == 'SASRec':
                pos_logits, neg_logits = model(u, seq, pos, neg)
            elif base == 'GRU4Rec':
                pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            loss.backward()
            
            adam_optimizer.step()

    sampler.close()

    torch.save(model.state_dict(), f'{name}.pt')

if __name__ == '__main__':
    train_and_save()