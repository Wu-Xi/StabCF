import torch
import torch.nn as nn
import pdb
import os
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,embedding0,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.embedding0 = embedding0
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        if self.embedding0:
            embs = [all_embed]
        else:
            embs = []

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns

        self.simi = args_config.simi
        self.warmup = args_config.warmup
        self.gamma = args_config.gamma

        self.p = args_config.p
        self.alpha = args_config.alpha
        self.beta = args_config.beta

        self.embedding0 = args_config.embedding0
        
        if self.ns == "rns":
            self.negative_sampling = self.rns_negative_sampling
        elif self.ns == "dns":
            self.negative_sampling = self.dns_negative_sampling
        elif self.ns == "mixgcf":
            self.negative_sampling = self.mixgcf_negative_sampling
        elif self.ns == "dins":
            self.negative_sampling = self.dins_negative_sampling
        elif self.ns == "dens":
            self.negative_sampling = self.dise_negative_sampling
        elif self.ns == "ahns":
            self.negative_sampling = self.adaptive_negative_sampling
        elif self.ns == "stabcf":
            self.negative_sampling = self.stabcf_negative_sampling
        else:
            raise NotImplementedError("Not found this negative sampling function...")  
        
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        
        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        
        self.sigmoid = nn.Sigmoid()
        
        print(f"This is {os.path.basename(__file__)} working...")
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         embedding0=self.embedding0,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def forward(self, batch=None,epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        if self.ns == 'stabcf':
            user_pos = batch['observed_pos_items']      # # [batch_size, window_length]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        pos_embs = item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.ns == 'dens':
            neg_user_embs = self.negative_sampling(epoch, user_gcn_emb, item_gcn_emb, user, neg_item[:, 0: self.n_negs], pos_item)
        elif self.ns == 'stabcf':
            pos_user_embs,neg_user_embs = self.negative_sampling(user_gcn_emb, item_gcn_emb, user, neg_item[:, 0: self.n_negs], pos_item, user_pos)
            pos_embs = pos_user_embs
        else:
            neg_user_embs = self.negative_sampling(user_gcn_emb, item_gcn_emb,user, neg_item[:, 0: self.n_negs],pos_item)
        batch_loss1,mf_loss1,emb_loss1=self.create_bpr_loss(user_gcn_emb[user], pos_embs, neg_user_embs)
        self._check_nan(batch_loss1)
    
        return batch_loss1,mf_loss1,emb_loss1
    
    def stabcf_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item, user_pos):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """Hard Positve Boundary Definition"""
        observed_pos_e = item_gcn_emb[user_pos]     # [batch_size, n_pos, n_hops+1, channel]
        observed_pos_score = (s_e.unsqueeze(dim=1) * observed_pos_e).sum(dim=-1)        # [batch_size, n_pos, n_hops+1]
        observed_pos_score = torch.exp(observed_pos_score)      # [batch_size, n_pos, n_hops+1]
              
        pos_score = (s_e * p_e).sum(dim=-1)     # [batch_size, n_hops+1]
        pos_score = self.alpha * torch.exp(pos_score) 
        
        # [batch_size, n_pos, n_hops+1] + [batch_size, n_hops+1]
        total_sum_pp = observed_pos_score.sum(dim=1) + pos_score      # [batch_size, n_hops+1]
        weight1 =  (observed_pos_score / total_sum_pp.unsqueeze(dim=1)).unsqueeze(dim=-1)      # [batch_size, items, channel, 1]
        weight2 = (pos_score / total_sum_pp).unsqueeze(dim=-1)      # [batch_size, channel]
        p_e_ = (weight1 * observed_pos_e).sum(dim=1) + weight2 * p_e        # [batch_size, channel]

        """Hard Negative Boundary Definition"""
        n_e = item_gcn_emb[neg_candidates]      # [batch_size, n_negs, n_hops+1, channel]
        scores_un = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)       #[batch_size, n_negs, n_hops+1]
        indices_max_un = torch.max(scores_un, dim=1)[1].detach()
        
        scores_pn = (p_e_.unsqueeze(dim=1) * n_e).sum(dim=-1)       #[batch_size, n_negs, n_hops+1]
        indices_max_pn = torch.max(scores_pn, dim=1)[1].detach()
        
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        
        neg_items_embedding_hardest_un = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_un, :] 
        neg_items_embedding_hardest_pn = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_pn, :] 
 
        """Importance-aware Mixup"""
        _pos_score = (s_e * p_e_).sum(dim=-1)       # [batch_size, n_hops+1]
        _pos_score = self.alpha * torch.exp(_pos_score)
        
        neg_score_un = (s_e * neg_items_embedding_hardest_un).sum(dim=-1)     # [batch_size, n_hops+1]
        neg_score_un = torch.exp(neg_score_un)
        
        neg_score_pn = (s_e * neg_items_embedding_hardest_pn).sum(dim=-1)     # [batch_size, n_hops+1]
        neg_score_pn = torch.exp(neg_score_pn)
        
        total_sum = neg_score_pn + neg_score_un + _pos_score      # [batch_size, n_hops+1]
        neg_weight_un = (neg_score_un / total_sum).unsqueeze(dim=-1)
        neg_weight_pn = (neg_score_pn / total_sum).unsqueeze(dim=-1)
        
        pos_weight = 1 - (neg_weight_un + neg_weight_pn)
        
        n_e_ =  pos_weight * p_e_ + neg_weight_un * neg_items_embedding_hardest_un + neg_weight_pn * neg_items_embedding_hardest_pn
        return p_e_, n_e_

    def dns_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e = user_gcn_emb[user] # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e)
            n_e_ = self.pooling(n_e.permute([0, 2, 1, 3]))   # # [batch_size, n_negs, channel]

        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs]
        indices = torch.max(scores, dim=1)[1].detach()  # torch.Size([2048, 3])
        return n_e[range(batch_size), indices, :]
       
    def mixgcf_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()        # torch.Size([2048, 4])
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

    def dins_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """Hard Boundary Definition"""
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()  # torch.Size([2048, 3])
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices, :]   #   [batch_size, n_hops+1, channel]

        """Dimension Independent Mixup"""
        neg_scores = torch.exp(s_e *neg_items_embedding_hardest)  # [batch_size, n_hops, channel]
        total_sum = self.alpha * torch.exp ((s_e * p_e))+neg_scores   # [batch_size, n_hops, channel]
        neg_weight = neg_scores/total_sum     # [batch_size, n_hops, channel]
        pos_weight = 1-neg_weight   # [batch_size, n_hops, channel]
        n_e_ =  pos_weight * p_e + neg_weight * neg_items_embedding_hardest  # mixing
        
        return n_e_

    def rns_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates,pos_item):
        neg_candidates = neg_candidates[:, 0:1]
        neg_candidates = neg_candidates.view(neg_candidates.shape[0])
        return item_gcn_emb[neg_candidates]
    
    def dise_negative_sampling(self, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        print(p_e.dtype, s_e.dtype)
        gate_p = torch.sigmoid(self.item_gate(p_e) + self.user_gate(s_e))
        gated_p_e = p_e * gate_p    # [batch_size, n_hops+1, channel]

        gate_n = torch.sigmoid(self.neg_gate(n_e) + self.pos_gate(gated_p_e).unsqueeze(1))
        gated_n_e = n_e * gate_n    # [batch_size, n_negs, n_hops+1, channel]
        
        n_e_sel = (1 - min(1, cur_epoch / self.warmup)) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - max(0, 1 - (cur_epoch / self.warmup))) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - self.alpha) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]
    
    def adaptive_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        
        s_e = s_e.mean(dim=1)  # [batch_size, channel]
        p_e = p_e.mean(dim=1)  # [batch_size, channel]
        n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]
                
        p_scores = self.similarity(s_e, p_e).unsqueeze(dim=1) # [batch_size, 1]
        n_scores = self.similarity(s_e.unsqueeze(dim=1), n_e) # [batch_size, n_negs]

        scores = torch.abs(n_scores - self.beta * (p_scores + self.alpha).pow(self.p + 1))

        """adaptive negative sampling"""
        indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()
        
        return item_gcn_emb[neg_item]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]
    
    def similarity(self, user_embeddings, item_embeddings):
        # [-1, n_hops, channel]
        if self.simi == 'ip':
            return (user_embeddings * item_embeddings).sum(dim=-1)
        elif self.simi == 'cos':
            return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
        elif self.simi == 'ed':
            return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
        else:  # ip
            return (user_embeddings * item_embeddings).sum(dim=-1)

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=-1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores)))       # [batch_size]

        if self.ns == 'dens' and self.gamma > 0.:
            gate_pos = torch.sigmoid(self.item_gate(pos_gcn_embs) + self.user_gate(user_gcn_emb))       # [batch_size, n_hops+1, channel]
            gated_pos_e_r = pos_gcn_embs * gate_pos
            gated_pos_e_ir = pos_gcn_embs - gated_pos_e_r
            gate_neg = torch.sigmoid(self.neg_gate(neg_gcn_embs) + self.pos_gate(gated_pos_e_r))

            gated_neg_e_r = neg_gcn_embs * gate_neg
            gated_neg_e_ir = neg_gcn_embs - gated_neg_e_r

            gated_pos_e_r = self.pooling(gated_pos_e_r)  # [batch_size, channel]
            gated_neg_e_r = self.pooling(gated_neg_e_r)  # [batch_size, channel]

            gated_pos_e_ir = self.pooling(gated_pos_e_ir)  # [batch_size, channel]
            gated_neg_e_ir = self.pooling(gated_neg_e_ir)  # [batch_size, channel]

            gated_pos_scores_r = torch.sum(torch.mul(u_e, gated_pos_e_r), axis=1)  # [batch_size]
            gated_neg_scores_r = torch.sum(torch.mul(u_e, gated_neg_e_r), axis=1)  # [batch_size]

            gated_pos_scores_ir = torch.sum(torch.mul(u_e, gated_pos_e_ir), axis=1)  # [batch_size]
            gated_neg_scores_ir = torch.sum(torch.mul(u_e, gated_neg_e_ir), axis=1)  # [batch_size]

            # BPR
            mf_loss += self.gamma * (
                torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_pos_scores_r))) +
                torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir))) +
                torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_pos_scores_r))) +
                torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_neg_scores_ir)))
            ) / 4

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:,0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * (regularize0) / batch_size     

        return mf_loss+emb_loss, mf_loss, emb_loss
    
# python main.py --gnn lightgcn --l2 0.001 --lr 0.001 --dataset ali --batch_size 2048 --gpu_id 0  --pool mean --n_negs 16 --alpha 2.0 