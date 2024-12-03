import numpy as np
import torch
import torch.nn as nn

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # (U, I)


class GRU4Rec(nn.Module):
    def __init__(
        self,
        device,
        hidden_size,
        embedding_dim,
        item_num,
        state_size,
        action_dim,
        gru_layers=1,
        use_packed_seq=True,
        train_pad_embed=True,
        padding_idx=0,
    ):
        super(GRU4Rec, self).__init__()
        self.dev=device
        self.layers = gru_layers
        self.hidden_dim = hidden_size
        self.embedding_dim = embedding_dim
        self.item_num = item_num
        self.state_size = state_size
        self.action_dim = action_dim
        self.use_packed_seq = use_packed_seq
        self.gru_layers = gru_layers

        # Use item num as default padding idx
        if padding_idx == None:
            padding_idx = self.item_num

        # Item-embeddings
        self.embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=None if train_pad_embed else padding_idx,
        )

        # Init strategy like in paper
        self.embedding.weight.data.normal_(mean=0, std=0.01)

        # Set padding embedding back to zero after init if its untrainable
        if not train_pad_embed:
            with torch.no_grad():
                self.embedding.weight[padding_idx] = torch.zeros(self.embedding_dim)

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
        )

        # Output layer
        self.output = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
        )

    def forward(self, seq, pos, neg):
        # s - (batch_size, n_memory)
        # lengths - (batch_size, true_state_len)
        # out - (batch_size, actions)

        seq_embs = self.embedding(torch.LongTensor(seq).to(self.dev))
        pos_embs = self.embedding(torch.LongTensor(pos).to(self.dev))
        neg_embs = self.embedding(torch.LongTensor(neg).to(self.dev))

        out, h = self.gru(seq_embs)

        out = self.output(out)

        pos_logits = (out * pos_embs).sum(dim=-1)
        neg_logits = (out * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits 

    def predict(self, seq, item_indices): # for inference
        seq_embs = self.embedding(torch.LongTensor(seq).to(self.dev))
        out, h = self.gru(seq_embs)
        out = self.output(out)

        final_feat = out[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.embedding(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
