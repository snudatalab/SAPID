import torch
import argparse
import numpy as np
import click

from model import SASRec, GRU4Rec
from util import *

'''
Generate a recommendation lists of SAPID from the given recommendation lists.
reclist : recommenation lists for all sessions
scrlist : recommendation scores of items in each recommendation list
item_prob : a popularity distribution of all items
candN : a size of a candidate pool
k : a length of a recommendation list
'''
def rerank(reclist, scrlist, item_prob, candN=20, k=10):
    cur_cnt = np.zeros(len(item_prob))
    left_slot = len(reclist)*k
    rtn = []
    for r, c in tqdm(zip(reclist, scrlist)):
        left_slot -= k
        exp_cnt = cur_cnt + left_slot * item_prob

        r = r[:candN]
        c = np.array(c[:candN])

        diff = c[:-1] - c[1:]
        thre = np.percentile(diff, 75)

        cutid = np.where(diff > thre)[0][-1]
        candlist = r[:cutid] if cutid > k else r[:k]

        cand_score = np.zeros(len(candlist))
        for idx, item in enumerate(candlist):
            cand_score[idx] = exp_cnt[item]
        rank = cand_score.argsort()
        rl = []
        for i in range(k):
            rl.append(rank[i])
            cur_cnt[candlist[rank[i]]] += 1
        rll = []
        for i in sorted(rl):
            rll.append(candlist[i])
        rtn.append(rll)
    return rtn


@click.command()
@click.option('--base', type=str, default='SASRec')
@click.option('--data', type=str, default='ml-1m')
@click.option('--epoch', type=int, default=200)
@click.option('--alpha', type=float, default=0.5)
@click.option('--candn', type=int, default=24)
@click.option('--path', type=str, default=None)
@click.option('--k', type=int, default=10)
def evaluate(base, data, epoch, alpha, candn, path, k):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read a dataset
    print('Read the dataset...')
    _, _, user_num, item_num, testset, _, item_prob, _, cnt = data_split_userseq(f'../data/{data}.pkl')

    # read parameters of a base model
    print('Load the model...')
    if base == 'SASRec' or base == 'sas':
        base = 'SASRec'
        args = argparse.Namespace(
            maxlen=200,
            hidden_units=50,
            num_blocks=2,
            num_heads=1,
            dropout_rate=0.5,
            device=device
        )
        if not path: path = f'../model/{data}/sas_alpha={alpha:.1f}_epoch={epoch}.pt'
        model = SASRec(user_num, item_num, args).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
    elif base == 'GRU4Rec' or base == 'gru':
        base = 'GRU4Rec'
        args = argparse.Namespace(
            hidden_units=50,
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
        if not path: path = f'../model/{data}/gru_alpha={alpha:.1f}_epoch={epoch}.pt'
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # generate recommendation lists of the base model
    print('Recommend...')
    rawreclist = []
    scrlist = []
    answer = []
    for x, y, ex in tqdm(testset):
        if base == 'SASRec':
            pred = -model.predict(*[np.array(l) for l in [[1], [x], [i for i in range(1, item_num+1)]]])
        elif base == 'GRU4Rec':
            pred = -model.predict(*[np.array(l) for l in [[x], [i for i in range(1, item_num+1)]]])
        
        pred = pred.squeeze()
        
        # do not recommend already interacted items
        for i in ex:
            pred[i-1] = 999999

        rawreclist.append((pred.argsort()+1).tolist()[:candn])
        scrlist.append((-pred[pred.argsort()[:candn]]).tolist())
        answer.append(y)
    
    # generate recommendation lists of SAPID
    print('Reranking...')
    reclist = rerank(rawreclist, scrlist, item_prob, candn, k=k)

    # evaludate the recommendation
    hr, ndcg, ent, gini = kfold_report_tuple(reclist, answer, user_num, item_num, k=k)

    print(f'HR@10 \t\t: {hr:.4f}')
    print(f'nDCG@10 \t: {ndcg:.4f}')
    print(f'Entropy@10 \t: {ent:.4f}')
    print(f'Gini@10 \t: {gini:.4f}')


if __name__ == '__main__':
    evaluate()