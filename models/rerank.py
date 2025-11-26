'''
Rerank models for fine-tuning stage
'''
import numpy as np
import torch
import torch.nn as nn
from models.layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention
from models.layers import Phi_function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class RerankBaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(RerankBaseModel, self).__init__()
        # Rerank阶段使用rerank任务逻辑
        # Rank和Rerank的模型处理逻辑相同，区别在于输入数据文件和候选数量
        self.task = 'rerank'
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len
        self.max_list_len = dataset.max_list_len

        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.rnn_dp = args.rnn_dp
        self.output_dim = args.output_dim
        self.convert_dropout = args.convert_dropout
        self.convert_type = args.convert_type
        self.auxiliary_loss_weight = args.auxi_loss_weight

        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim
        self.dens_vec_num = 0

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim)
        self.attr_embedding = nn.Embedding(self.attr_num + 1, self.embed_dim)
        self.rating_embedding = nn.Embedding(self.rating_num + 1, self.embed_dim)
        if self.augment_num:
            aug_vec_dim = dataset.aug_vec_dim if hasattr(dataset, 'aug_vec_dim') and dataset.aug_vec_dim > 0 else 768
            
            # 🔧 知识降维模块 - 针对小数据集优化
            self.enable_knowledge_reduction = getattr(args, 'enable_knowledge_reduction', False)
            if self.enable_knowledge_reduction:
                reduction_dim = getattr(args, 'knowledge_reduction_dim', 128)
                reduction_dropout = getattr(args, 'knowledge_reduction_dropout', 0.3)
                
                # 降维层: 768 → reduction_dim (默认128)
                self.knowledge_reducer = nn.Sequential(
                    nn.Linear(aug_vec_dim, reduction_dim),
                    nn.BatchNorm1d(reduction_dim),
                    nn.ReLU(),
                    nn.Dropout(reduction_dropout)
                )
                # 使用降维后的维度
                convert_input_dim = reduction_dim
                print(f"✅ 启用知识降维: {aug_vec_dim}维 → {reduction_dim}维 (dropout={reduction_dropout})")
            else:
                self.knowledge_reducer = None
                convert_input_dim = aug_vec_dim
            
            self.convert_module = ConvertNet(args, convert_input_dim, self.convert_dropout, self.convert_type)
            self.dens_vec_num = args.convert_arch[-1] * self.augment_num

        self.module_inp_dim = self.get_input_dim()
        self.field_num = self.get_field_num()
        self.convert_loss = 0

    def process_input(self, inp):
        device = next(self.parameters()).device
        hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
        hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                 self.embed_dim * self.attr_fnum)
        hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                      self.embed_dim)
        hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
        hist_len = inp['hist_seq_len'].to(device)

        # Rerank任务逻辑
        if self.task == 'rerank':
            iid_emb = self.item_embedding(inp['iid_list'].to(device))
            attr_emb = self.attr_embedding(inp['aid_list'].to(device)).view(-1, self.max_list_len,
                                                                            self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            item_emb = item_emb.view(-1, self.max_list_len, self.itm_emb_dim)
            labels = inp['lb_list'].to(device).view(-1, self.max_list_len)
            if self.augment_num:
                hist_aug = inp['hist_aug_vec'].to(device)
                item_list_aug = inp['item_aug_vec_list']
                
                # 🔧 应用知识降维
                if self.knowledge_reducer is not None:
                    hist_aug = self.knowledge_reducer(hist_aug)
                    item_list_aug = [self.knowledge_reducer(item_aug.to(device)) for item_aug in item_list_aug]
                    orig_dens_list = [[hist_aug, item_aug] for item_aug in item_list_aug]
                else:
                    orig_dens_list = [[hist_aug, item_aug.to(device)] for item_aug in item_list_aug]
                
                # 处理长短兴趣比例信息（重排序任务）
                ls_ratios_list = None
                if hasattr(self.args, 'enable_ls_attention') and self.args.enable_ls_attention:
                    if 'short_term_ratio' in inp and 'long_term_ratio' in inp:
                        short_ratios = inp['short_term_ratio'].to(device)
                        long_ratios = inp['long_term_ratio'].to(device)
                        ls_ratios_list = [[(short_ratios, long_ratios), (short_ratios, long_ratios)] for _ in range(len(orig_dens_list))]
                
                if ls_ratios_list:
                    dens_vec_list = [self.convert_module(orig_dens, ls_ratios) for orig_dens, ls_ratios in zip(orig_dens_list, ls_ratios_list)]
                else:
                    dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                dens_vec = torch.stack([dens for dens in dens_vec_list], dim=1)
            else:
                dens_vec, orig_dens_list = None, None

            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_list, labels
        else:
            raise NotImplementedError

    def get_input_dim(self):
        return self.itm_emb_dim + self.dens_vec_num

    def get_field_num(self):
        return self.item_fnum + self.augment_num + self.hist_fnum

    def process_rerank_inp(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            out = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            out = item_embedding
        return out, labels

    def get_rerank_output(self, logits, labels=None, attn=False):
        outputs = {
            'logits': logits,
            'labels': labels,
        }

        if labels is not None:
            if attn:
                logits = attention_score(logits.view(-1, self.max_list_len), self.args.temperature)
                labels = attention_score(labels.float().view(-1, self.max_list_len), self.args.temperature)
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight
        return outputs

    def get_mask(self, length, max_len):
        device = next(self.parameters()).device
        rang = torch.arange(0, max_len).view(-1, max_len).to(device)
        batch_rang = rang.repeat([length.shape[0], 1])
        mask = batch_rang < torch.unsqueeze(length, dim=-1)
        return mask.unsqueeze(dim=-1).long()

class DLCM(RerankBaseModel):
    def __init__(self, args, dataset):
        super(DLCM, self).__init__(args, dataset)
        self.gru = torch.nn.GRU(self.module_inp_dim, self.hidden_size, dropout=self.rnn_dp, batch_first=True)
        self.phi_function = Phi_function(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        seq_state, final_state = self.gru(processed_inp)
        final_state = torch.squeeze(final_state, dim=0)

        scores = self.phi_function(seq_state, final_state)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class SetRank(RerankBaseModel):
    def __init__(self, args, dataset):
        super(SetRank, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels, attn=True)
        return outputs


class GSF(RerankBaseModel):
    """
    Groupwise Scoring Function (GSF) for Reranking
    Groups items and applies position-aware scoring for better list-wise optimization
    """
    def __init__(self, args, dataset):
        super(GSF, self).__init__(args, dataset)
        self.group_size = getattr(args, 'group_size', 3)
        
        # Position embeddings for groupwise modeling
        self.pos_embedding = nn.Embedding(self.max_list_len, self.embed_dim)
        
        # Group-level attention
        self.group_attn = nn.MultiheadAttention(
            self.module_inp_dim + self.embed_dim, 
            args.n_head, 
            batch_first=True,
            dropout=args.attn_dp
        )
        
        # Scoring network
        self.score_mlp = MLP(args.final_mlp_arch, self.module_inp_dim + self.embed_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)
        
    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        batch_size = item_embed.shape[0]
        
        # Add position embeddings
        positions = torch.arange(self.max_list_len, device=item_embed.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.pos_embedding(positions)
        item_embed_pos = torch.cat([item_embed, pos_emb], dim=-1)
        
        # Apply group-level attention
        attn_out, _ = self.group_attn(item_embed_pos, item_embed_pos, item_embed_pos)
        
        # Score computation
        mlp_out = self.score_mlp(attn_out)
        scores = self.fc_out(mlp_out).view(-1, self.max_list_len)
        scores = torch.sigmoid(scores)
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class EGRerank(RerankBaseModel):
    """
    Expected Gain Reranking
    Optimizes expected utility by considering both relevance and diversity
    """
    def __init__(self, args, dataset):
        super(EGRerank, self).__init__(args, dataset)
        
        self.diversity_weight = getattr(args, 'diversity_weight', 0.1)
        
        # Relevance scoring network
        self.relevance_attn = nn.MultiheadAttention(
            self.module_inp_dim, 
            args.n_head, 
            batch_first=True,
            dropout=args.attn_dp
        )
        self.relevance_mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.relevance_fc = nn.Linear(args.final_mlp_arch[-1], 1)
        
        # Diversity scoring network
        self.diversity_fc = nn.Linear(self.module_inp_dim, self.embed_dim)
        
    def compute_diversity(self, item_embed):
        """
        Compute pairwise diversity among items
        """
        # Project to diversity space
        diversity_repr = self.diversity_fc(item_embed)
        
        # Compute pairwise cosine similarity
        normalized = torch.nn.functional.normalize(diversity_repr, p=2, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        
        # Diversity is inverse of similarity (excluding diagonal)
        mask = torch.eye(self.max_list_len, device=item_embed.device).unsqueeze(0)
        diversity = 1.0 - (similarity * (1 - mask)).sum(dim=-1) / (self.max_list_len - 1)
        
        return diversity
        
    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        
        # Compute relevance scores
        attn_out, _ = self.relevance_attn(item_embed, item_embed, item_embed)
        mlp_inp = torch.cat([item_embed, attn_out], dim=-1)
        mlp_out = self.relevance_mlp(mlp_inp)
        relevance_scores = torch.sigmoid(self.relevance_fc(mlp_out)).squeeze(-1)
        
        # Compute diversity scores
        diversity_scores = self.compute_diversity(item_embed)
        
        # Combine relevance and diversity
        scores = relevance_scores + self.diversity_weight * diversity_scores
        
        outputs = self.get_rerank_output(scores, labels)
        return outputs
