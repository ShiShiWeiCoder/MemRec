'''
Rank models for coarse-ranking stage
'''
import numpy as np
import torch
import torch.nn as nn
from models.layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class RankBaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(RankBaseModel, self).__init__()
        # Rank阶段使用rerank任务逻辑，区别在于输入数据文件（rank.train vs rerank.train）
        self.task = 'rerank'  # Rank模型使用rerank任务逻辑
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

        # Rank模型使用rerank任务逻辑
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


class DeepInterestNet(RankBaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DeepInterestNet, self).__init__(args, dataset)

        self.map_layer = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
        # embedding of history item and candidate item should be the same
        self.attention_net = AttentionPoolingLayer(self.itm_emb_dim, self.dropout)

        # history embedding, item embedding, and user embedding
        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            batch_size = query.shape[0]
            max_list_len = query.shape[1]
            mask = self.get_mask(hist_len, self.max_hist_len)
            
            user_behavior = self.map_layer(user_behavior)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = query[:, i, :]  # (batch_size, item_emb_dim)
                # AttentionPoolingLayer期望query是(batch_size, dim)，不是(batch_size, 1, dim)
                user_interest, _ = self.attention_net(item_emb, user_behavior, mask)  # (batch_size, item_emb_dim)
                
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    concat_input = torch.cat([user_interest, item_emb, dens_emb], dim=-1)
                else:
                    concat_input = torch.cat([user_interest, item_emb], dim=-1)
                
                mlp_out = self.final_mlp(concat_input)
                logit = self.final_fc(mlp_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            out = self.get_rerank_output(scores, labels)
            return out


class DIEN(RankBaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DIEN, self).__init__(args, dataset)

        self.interest_extractor = InterestExtractor(self.hist_emb_dim, self.itm_emb_dim)
        self.interest_evolution = InterestEvolving(self.itm_emb_dim, gru_type=args.dien_gru, dropout=self.dropout)

        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, length, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            batch_size = query.shape[0]
            max_list_len = query.shape[1]
            mask = self.get_mask(length, self.max_hist_len)
            length_expanded = torch.unsqueeze(length, dim=-1)
            
            masked_interest = self.interest_extractor(user_behavior, length_expanded)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = query[:, i, :]  # (batch_size, item_emb_dim)
                user_interest = self.interest_evolution(item_emb, masked_interest, length_expanded, mask)
                
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    concat_input = torch.cat([user_interest, item_emb, dens_emb], dim=-1)
                else:
                    concat_input = torch.cat([user_interest, item_emb], dim=-1)
                
                mlp_out = self.final_mlp(concat_input)
                logit = self.final_fc(mlp_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            out = self.get_rerank_output(scores, labels)
            return out


class DCN(RankBaseModel):
    '''
    DCNv1
    '''
    def __init__(self, args, mode, dataset):
        super(DCN, self).__init__(args, dataset)
        self.deep_arch = args.dcn_deep_arch
        self.cross_net = CrossNet(self.module_inp_dim, args.dcn_cross_num, mode)
        self.deep_net = MLP(self.deep_arch, self.module_inp_dim, self.dropout)
        final_inp_dim = self.module_inp_dim + self.deep_arch[-1]
        self.final_mlp = MLP(self.final_mlp_arch, final_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)
        
        # 对于rerank任务，需要单独的网络（输入包含user history）
        if self.task == 'rerank':
            # rerank任务的输入维度：itm_emb_dim + hist_emb_dim + dens_vec_num
            rerank_input_dim = self.itm_emb_dim + self.hist_emb_dim + self.dens_vec_num
            self.cross_net_rerank = CrossNet(rerank_input_dim, args.dcn_cross_num, mode)
            self.deep_net_rerank = MLP(self.deep_arch, rerank_input_dim, self.dropout)
            rerank_final_inp_dim = rerank_input_dim + self.deep_arch[-1]
            self.final_mlp_rerank = MLP(self.final_mlp_arch, rerank_final_inp_dim, self.dropout)
            self.final_fc_rerank = nn.Linear(self.final_mlp_arch[-1], 1)


    def forward(self, inp):
        '''
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        '''
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]
            
            # 将user_behavior平均池化得到用户表示
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)
                
                # 构建输入
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    model_inp = torch.cat([item_emb, user_repr, dens_emb], dim=1)
                else:
                    model_inp = torch.cat([item_emb, user_repr], dim=1)
                
                # Deep部分和Cross部分（使用rerank专用网络）
                deep_part = self.deep_net_rerank(model_inp)
                cross_part = self.cross_net_rerank(model_inp)
                
                # 合并并通过最终MLP
                final_inp = torch.cat([deep_part, cross_part], dim=1)
                mlp_out = self.final_mlp_rerank(final_inp)
                logit = self.final_fc_rerank(mlp_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs


class DeepFM(RankBaseModel):
    def __init__(self, args, dataset):
        super(DeepFM, self).__init__(args, dataset)
        # FM
        self.fm_first_iid_emb = nn.Embedding(self.item_num + 1, 1)
        self.fm_first_aid_emb = nn.Embedding(self.attr_num + 1, 1)
        self.fm_first_dense_weight = nn.Parameter(torch.rand([self.dens_vec_num, 1]))
        # DNN
        self.deep_part = MLP(args.deepfm_deep_arch, self.module_inp_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.deepfm_deep_arch[-1], 1)
        
        # 对于rerank任务，需要额外的投影层来处理user history
        if self.task == 'rerank':
            # rerank任务中，需要将user history投影到与item相同的空间
            self.user_proj = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
            # rerank任务的DNN输入维度：itm_emb_dim (item) + itm_emb_dim (projected user) + dens_vec_num
            rerank_dnn_input_dim = self.itm_emb_dim * 2 + self.dens_vec_num
            self.deep_part_rerank = MLP(args.deepfm_deep_arch, rerank_dnn_input_dim, self.dropout)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        device = next(self.parameters()).device
        
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]
            
            # 将user_behavior投影到与item_embedding相同的维度
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)
            user_repr_proj = self.user_proj(user_repr)  # (batch_size, itm_emb_dim)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)
                
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec  # (batch_size, dens_dim)
                    dnn_inp = torch.cat([item_emb, user_repr_proj, dens_emb], dim=1)
                else:
                    dnn_inp = torch.cat([item_emb, user_repr_proj], dim=1)
                
                # FM first order (简化版，rerank任务不使用原始iid/aid)
                fm_logit = torch.zeros(batch_size, 1, device=device)
                
                # FM second order - 将item和投影后的user嵌入分别reshape为field形式
                # item_emb: (batch_size, itm_emb_dim) -> reshape为 (batch_size, item_fnum, embed_dim)
                item_emb_field = item_emb.view(batch_size, self.item_fnum, self.embed_dim)
                # user_repr_proj: (batch_size, itm_emb_dim) -> reshape为 (batch_size, item_fnum, embed_dim)
                user_emb_field = user_repr_proj.view(batch_size, self.item_fnum, self.embed_dim)
                
                # 拼接所有fields
                fm_second_inp = torch.cat([item_emb_field, user_emb_field], dim=1)  # (batch_size, item_fnum*2, embed_dim)
                
                square_of_sum = torch.pow(torch.sum(fm_second_inp, dim=1, keepdim=True), 2)
                sum_of_square = torch.sum(torch.pow(fm_second_inp, 2), dim=1, keepdim=True)
                cross_term = square_of_sum - sum_of_square
                cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
                fm_logit += cross_term
                
                # DNN - 使用专门为rerank任务创建的MLP
                deep_out = self.deep_part_rerank(dnn_inp)
                logit = fm_logit + self.dnn_fc_out(deep_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs


class AutoInt(RankBaseModel):
    def __init__(self, args, dataset):
        super(AutoInt, self).__init__(args, dataset)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(args.embed_dim if i == 0 else args.num_attn_heads * args.attn_size,
                                     attention_dim=args.attn_size,
                                     num_heads=args.num_attn_heads,
                                     dropout_rate=args.dropout,
                                     use_residual=args.res_conn,
                                     use_scale=args.attn_scale,
                                     layer_norm=False,
                                     align_to='output')
              for i in range(args.num_attn_layers)])
        final_dim = self.field_num * args.attn_size * args.num_attn_heads

        self.attn_out = nn.Linear(final_dim, 1)

    def forward(self, inp):
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]
            
            # 获取用户表示
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)
                
                # 构建输入
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    field_inp = torch.cat([item_emb, user_repr, dens_emb], dim=1)
                else:
                    field_inp = torch.cat([item_emb, user_repr], dim=1)
                
                # 转换为field表示
                field_inp = field_inp.view(batch_size, self.field_num, self.embed_dim)
                
                # 通过self-attention
                attention_out = self.self_attention(field_inp)
                attention_out = torch.flatten(attention_out, start_dim=1)
                
                logit = self.attn_out(attention_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs


class FiBiNet(RankBaseModel):
    def __init__(self, args, dataset):
        super(FiBiNet, self).__init__(args, dataset)
        self.senet_layer = SqueezeExtractionLayer(self.field_num, args.reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(self.embed_dim, self.field_num, args.bilinear_type)
        final_dim = self.field_num * (self.field_num - 1) * self.embed_dim
        self.dnn = MLP(args.final_mlp_arch, final_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]
            
            # 获取用户表示
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)
                
                # 构建输入
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    field_inp = torch.cat([item_emb, user_repr, dens_emb], dim=1)
                else:
                    field_inp = torch.cat([item_emb, user_repr], dim=1)
                
                # 转换为field表示
                feat_embed = field_inp.view(batch_size, self.field_num, self.embed_dim)
                
                # FiBiNet逻辑
                senet_embed = self.senet_layer(feat_embed)
                bilinear_p = self.bilinear_layer(feat_embed)
                bilinear_q = self.bilinear_layer(senet_embed)
                comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
                
                logit = self.fc_out(self.dnn(comb_out))
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs


class FiGNN(RankBaseModel):
    def __init__(self, args, dataset):
        super(FiGNN, self).__init__(args, dataset)
        self.fignn = FiGNNBlock(self.field_num, self.embed_dim, args.gnn_layer_num,
                                args.res_conn, args.reuse_graph_layer)
        self.fc = AttentionalPrediction(self.field_num, self.embed_dim)

    def forward(self, inp):
        if self.task == 'rerank':
            # Rerank任务：处理多个候选items
            item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]
            
            # 获取用户表示
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)
            
            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)
                
                # 构建输入
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :] if len(dens_vec.shape) == 3 else dens_vec
                    field_inp = torch.cat([item_emb, user_repr, dens_emb], dim=1)
                else:
                    field_inp = torch.cat([item_emb, user_repr], dim=1)
                
                # 转换为field表示
                feat_embed = field_inp.view(batch_size, self.field_num, self.embed_dim)
                
                # FiGNN逻辑
                h_out = self.fignn(feat_embed)
                logit = self.fc(h_out)
                scores.append(logit)
            
            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs
