'''
-*- coding: utf-8 -*-
@File  : models.py
'''
import numpy as np
import torch
import torch.nn as nn
from layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention
from layers import Phi_function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class BaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()
        # task参数: 'ctr'表示单item预测，'rerank'表示多候选排序（包括Rank粗排和Rerank精排）
        # Rank和Rerank的模型处理逻辑相同，区别在于输入数据文件和候选数量
        self.task = args.task
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len
        if self.task == 'rerank' or self.task == 'rank':
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
            analysis_vec_dim = getattr(args, 'analysis_vec_dim', aug_vec_dim)

            self.enable_knowledge_reduction = getattr(args, 'enable_knowledge_reduction', False)
            self.unified_reduction = getattr(args, 'unified_reduction', False)
            reduction_dim = getattr(args, 'knowledge_reduction_dim', 128)
            reduction_dropout = getattr(args, 'knowledge_reduction_dropout', 0.3)

            if self.unified_reduction and aug_vec_dim == analysis_vec_dim:
                self.knowledge_reducer = nn.Sequential(
                    nn.Linear(aug_vec_dim, reduction_dim),
                    nn.LayerNorm(reduction_dim),
                    nn.GELU(),
                    nn.Dropout(reduction_dropout)
                )
                self.analysis_reducer = None
                convert_input_dim = reduction_dim
                args.analysis_dim_after_reduction = reduction_dim
                print(f"✅ 统一降维 (三向量共享): {aug_vec_dim}维 → {reduction_dim}维")
            elif self.enable_knowledge_reduction:
                self.knowledge_reducer = nn.Sequential(
                    nn.Linear(aug_vec_dim, reduction_dim),
                    nn.BatchNorm1d(reduction_dim),
                    nn.ReLU(),
                    nn.Dropout(reduction_dropout)
                )
                convert_input_dim = reduction_dim
                print(f"✅ 启用知识降维: {aug_vec_dim}维 → {reduction_dim}维")
                analysis_reduction_dim = getattr(args, 'analysis_reduction_dim', 0)
                if analysis_reduction_dim > 0:
                    self.analysis_reducer = nn.Sequential(
                        nn.Linear(analysis_vec_dim, analysis_reduction_dim),
                        nn.LayerNorm(analysis_reduction_dim),
                        nn.GELU(),
                        nn.Dropout(reduction_dropout)
                    )
                    args.analysis_dim_after_reduction = analysis_reduction_dim
                    print(f"✅ 独立分析降维: {analysis_vec_dim}维 → {analysis_reduction_dim}维")
                else:
                    self.analysis_reducer = None
                    args.analysis_dim_after_reduction = analysis_vec_dim
            else:
                self.knowledge_reducer = None
                convert_input_dim = aug_vec_dim
                analysis_reduction_dim = getattr(args, 'analysis_reduction_dim', 0)
                if analysis_reduction_dim > 0:
                    self.analysis_reducer = nn.Sequential(
                        nn.Linear(analysis_vec_dim, analysis_reduction_dim),
                        nn.LayerNorm(analysis_reduction_dim),
                        nn.GELU(),
                        nn.Dropout(reduction_dropout)
                    )
                    args.analysis_dim_after_reduction = analysis_reduction_dim
                    print(f"✅ 独立分析降维: {analysis_vec_dim}维 → {analysis_reduction_dim}维")
                else:
                    self.analysis_reducer = None
                    args.analysis_dim_after_reduction = analysis_vec_dim

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

        if self.task == 'ctr':
            iid_emb = self.item_embedding(inp['iid'].to(device))
            attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
                        # item_emb = item_emb.view(-1, self.itm_emb_dim)
            labels = inp['lb'].to(device)
            if self.augment_num:
                hist_aug_vec = inp['hist_aug_vec'].to(device)
                item_aug_vec = inp['item_aug_vec'].to(device)
                if getattr(self.args, 'skip_user_profile', False):
                    hist_aug_vec = torch.zeros_like(hist_aug_vec)
                if getattr(self.args, 'skip_course_profile', False):
                    item_aug_vec = torch.zeros_like(item_aug_vec)

                # 🔧 应用知识降维
                if self.knowledge_reducer is not None:
                    batch_size = hist_aug_vec.shape[0]
                    hist_aug_vec = self.knowledge_reducer(hist_aug_vec)
                    item_aug_vec = self.knowledge_reducer(item_aug_vec)

                orig_dens_vec = [hist_aug_vec, item_aug_vec]

                # 🧠 处理多级记忆数据
                multilevel_memory_data = None
                if ('multilevel_memory_mode' in inp or
                    hasattr(self.args, 'memory_mode') and getattr(self.args, 'memory_mode', False)):

                    # 构建多级记忆数据字典
                    multilevel_memory_data = {}

                    # 添加三级记忆向量
                    for memory_type in ['sensory_memory_vec', 'working_memory_vec', 'longterm_memory_vec']:
                        if memory_type in inp:
                            multilevel_memory_data[memory_type] = inp[memory_type].to(device)

                    # 添加三级记忆比例
                    for ratio_type in ['sensory_memory_ratio', 'working_memory_ratio', 'longterm_memory_ratio']:
                        if ratio_type in inp:
                            multilevel_memory_data[ratio_type] = inp[ratio_type].to(device)

                    skip_analysis = getattr(self.args, 'skip_analysis', False)
                    if 'memory_analysis_aug_vec' in inp and not skip_analysis:
                        av = inp['memory_analysis_aug_vec'].to(device)
                        if self.analysis_reducer is not None:
                            av = self.analysis_reducer(av)
                        elif self.knowledge_reducer is not None:
                            av = self.knowledge_reducer(av)
                        orig_dens_vec.append(av)

                    if 'enhanced_gating_features' in inp:
                        multilevel_memory_data['enhanced_gating_features'] = inp['enhanced_gating_features'].to(device)
                    if 'transition_features' in inp:
                        multilevel_memory_data['transition_features'] = inp['transition_features'].to(device)

                    dens_vec = self.convert_module(orig_dens_vec, multilevel_memory_data=multilevel_memory_data)

                # 处理长短兴趣比例信息（兼容性保持）
                elif hasattr(self.args, 'enable_ls_attention') and self.args.enable_ls_attention:
                    ls_ratios = None
                    if 'short_term_ratio' in inp and 'long_term_ratio' in inp:
                        short_ratios = inp['short_term_ratio'].to(device)
                        long_ratios = inp['long_term_ratio'].to(device)
                        # 为每个增强向量提供比例信息
                        ls_ratios = [(short_ratios, long_ratios), (short_ratios, long_ratios)]
                        if 'analysis_aug_vec' in inp:
                            orig_dens_vec.append(inp['analysis_aug_vec'].to(device))
                            ls_ratios.append((short_ratios, long_ratios))

                    dens_vec = self.convert_module(orig_dens_vec, ls_ratios)
                else:
                    # 默认处理（无增强）
                    dens_vec = self.convert_module(orig_dens_vec)
            else:
                dens_vec, orig_dens_vec = None, None
            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_vec, labels
        elif self.task == 'rerank' or self.task == 'rank':
            iid_emb = self.item_embedding(inp['iid_list'].to(device))
            attr_emb = self.attr_embedding(inp['aid_list'].to(device)).view(-1, self.max_list_len,
                                                                            self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            item_emb = item_emb.view(-1, self.max_list_len, self.itm_emb_dim)
            labels = inp['lb_list'].to(device).view(-1, self.max_list_len)
            if self.augment_num:
                hist_aug = inp['hist_aug_vec'].to(device)
                item_list_aug = inp['item_aug_vec_list']
                if getattr(self.args, 'skip_user_profile', False):
                    hist_aug = torch.zeros_like(hist_aug)
                if getattr(self.args, 'skip_course_profile', False):
                    item_list_aug = [torch.zeros_like(item_aug.to(device)) for item_aug in item_list_aug]

                # 🔧 应用知识降维
                if self.knowledge_reducer is not None:
                    hist_aug = self.knowledge_reducer(hist_aug)
                    item_list_aug = [self.knowledge_reducer(item_aug.to(device)) for item_aug in item_list_aug]
                    orig_dens_list = [[hist_aug, item_aug] for item_aug in item_list_aug]
                else:
                    orig_dens_list = [[hist_aug, item_aug.to(device)] for item_aug in item_list_aug]

                # 🧠 处理多级记忆数据（rerank/rank 任务）
                multilevel_memory_data = None
                if ('multilevel_memory_mode' in inp or
                    hasattr(self.args, 'memory_mode') and getattr(self.args, 'memory_mode', False)):

                    multilevel_memory_data = {}

                    for memory_type in ['sensory_memory_vec', 'working_memory_vec', 'longterm_memory_vec']:
                        if memory_type in inp:
                            multilevel_memory_data[memory_type] = inp[memory_type].to(device)

                    for ratio_type in ['sensory_memory_ratio', 'working_memory_ratio', 'longterm_memory_ratio']:
                        if ratio_type in inp:
                            multilevel_memory_data[ratio_type] = inp[ratio_type].to(device)

                    skip_analysis = getattr(self.args, 'skip_analysis', False)
                    if 'memory_analysis_aug_vec' in inp and not skip_analysis:
                        analysis_vec = inp['memory_analysis_aug_vec'].to(device)
                        if self.analysis_reducer is not None:
                            analysis_vec = self.analysis_reducer(analysis_vec)
                        elif self.knowledge_reducer is not None:
                            analysis_vec = self.knowledge_reducer(analysis_vec)
                        orig_dens_list = [[hist_aug, item_aug, analysis_vec] for _, [hist_aug, item_aug] in enumerate(orig_dens_list)]

                    if 'enhanced_gating_features' in inp:
                        multilevel_memory_data['enhanced_gating_features'] = inp['enhanced_gating_features'].to(device)
                    if 'transition_features' in inp:
                        multilevel_memory_data['transition_features'] = inp['transition_features'].to(device)

                    # 批量化处理：将所有候选合并为一个大 batch 调用一次 convert_module
                    num_candidates = len(orig_dens_list)
                    batch_size = hist_aug.size(0)
                    num_vecs = len(orig_dens_list[0])

                    batched_vecs = []
                    for vec_idx in range(num_vecs):
                        stacked = torch.stack([orig_dens_list[c][vec_idx] for c in range(num_candidates)], dim=1)  # (B, C, D)
                        batched_vecs.append(stacked.reshape(batch_size * num_candidates, -1))  # (B*C, D)

                    batched_memory_data = {}
                    for k, v in multilevel_memory_data.items():
                        if isinstance(v, torch.Tensor):
                            batched_memory_data[k] = v.unsqueeze(1).expand(-1, num_candidates, *v.shape[1:]).reshape(batch_size * num_candidates, *v.shape[1:])
                        else:
                            batched_memory_data[k] = v

                    batched_out = self.convert_module(batched_vecs, multilevel_memory_data=batched_memory_data)
                    dens_vec = batched_out.reshape(batch_size, num_candidates, -1)

                # 处理长短兴趣比例信息（重排序任务）
                elif hasattr(self.args, 'enable_ls_attention') and self.args.enable_ls_attention:
                    ls_ratios_list = None
                    if 'short_term_ratio' in inp and 'long_term_ratio' in inp:
                        short_ratios = inp['short_term_ratio'].to(device)
                        long_ratios = inp['long_term_ratio'].to(device)
                        ls_ratios_list = [[(short_ratios, long_ratios), (short_ratios, long_ratios)] for _ in range(len(orig_dens_list))]

                    if ls_ratios_list:
                        dens_vec_list = [self.convert_module(orig_dens, ls_ratios) for orig_dens, ls_ratios in zip(orig_dens_list, ls_ratios_list)]
                    else:
                        dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                    dens_vec = torch.stack(dens_vec_list, dim=1)
                else:
                    dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                    dens_vec = torch.stack(dens_vec_list, dim=1)
            else:
                dens_vec, orig_dens_list = None, None

            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_list, labels
        else:
            raise NotImplementedError

    def get_input_dim(self):
        if self.task == 'ctr':
            return self.hist_emb_dim + self.itm_emb_dim + self.dens_vec_num
        elif self.task == 'rerank' or self.task == 'rank':
            # rank和rerank任务使用相同的输入维度计算
            return self.itm_emb_dim + self.dens_vec_num
        else:
            raise NotImplementedError

    def get_field_num(self):
        return self.item_fnum + self.augment_num + self.hist_fnum

    def get_filed_input(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels

    def process_rerank_inp(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            out = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            out = item_embedding
        return out, labels

    def get_ctr_output(self, logits, labels=None):
        outputs = {
            'logits': torch.sigmoid(logits),
            'labels': labels,
        }

        if labels is not None:
            if self.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.output_dim)), labels.float())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight

        return outputs

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


class DeepInterestNet(BaseModel):
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

        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            mask = self.get_mask(hist_len, self.max_hist_len)
            user_behavior = self.map_layer(user_behavior)
            user_interest, _ = self.attention_net(query, user_behavior, mask)

            if self.augment_num:
                concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
            else:
                concat_input = torch.cat([user_interest, query], dim=-1)

            mlp_out = self.final_mlp(concat_input)
            logits = self.final_fc(mlp_out)
            out = self.get_ctr_output(logits, labels)
            return out


class DIEN(BaseModel):
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

        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            mask = self.get_mask(length, self.max_hist_len)
            length = torch.unsqueeze(length, dim=-1)
            masked_interest = self.interest_extractor(user_behavior, length)
            user_interest = self.interest_evolution(query, masked_interest, length, mask)

            if self.augment_num:
                concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
            else:
                concat_input = torch.cat([user_interest, query], dim=-1)

            mlp_out = self.final_mlp(concat_input)
            logits = self.final_fc(mlp_out)
            out = self.get_ctr_output(logits, labels)
            return out


class DCN(BaseModel):
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
        if self.task == 'rerank' or self.task == 'rank':
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

        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
            if self.augment_num:
                model_inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
            else:
                model_inp = torch.cat([item_embedding, user_behavior], dim=1)

            deep_part = self.deep_net(model_inp)
            cross_part = self.cross_net(model_inp)

            final_inp = torch.cat([deep_part, cross_part], dim=1)
            mlp_out = self.final_mlp(final_inp)
            logits = self.final_fc(mlp_out)
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class DeepFM(BaseModel):
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
        if self.task == 'rerank' or self.task == 'rank':
            # rerank任务中，需要将user history投影到与item相同的空间
            self.user_proj = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
            # rerank任务的DNN输入维度：itm_emb_dim (item) + itm_emb_dim (projected user) + dens_vec_num
            rerank_dnn_input_dim = self.itm_emb_dim * 2 + self.dens_vec_num
            self.deep_part_rerank = MLP(args.deepfm_deep_arch, rerank_dnn_input_dim, self.dropout)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        device = next(self.parameters()).device

        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
            if self.augment_num:
                dnn_inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
            else:
                dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

            # fm first order
            iid_first = self.fm_first_iid_emb(inp['iid'].to(device)).view(-1, 1)
            aid_first = self.fm_first_aid_emb(inp['aid'].to(device)).view(-1, self.attr_fnum)
            linear_sparse_logit = torch.sum(torch.cat([iid_first, aid_first], dim=1), dim=1).view(-1, 1)
            if self.augment_num:
                linear_dense_logit = dens_vec.matmul(self.fm_first_dense_weight).view(-1, 1)
                fm_logit = linear_sparse_logit + linear_dense_logit
            else:
                fm_logit = linear_sparse_logit

            # fm second order
            fm_second_inp = torch.cat([item_embedding, user_behavior], dim=1)
            fm_second_inp = fm_second_inp.view(-1, self.item_fnum + self.hist_fnum, self.embed_dim)

            square_of_sum = torch.pow(torch.sum(fm_second_inp, dim=1, keepdim=True), 2)
            sum_of_square = torch.sum(torch.pow(fm_second_inp, 2), dim=1, keepdim=True)
            cross_term = square_of_sum - sum_of_square
            cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
            fm_logit += cross_term

            # dnn
            deep_out = self.deep_part(dnn_inp)
            logits = fm_logit + self.dnn_fc_out(deep_out)
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class xDeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(xDeepFM, self).__init__(args, dataset)
        input_dim = self.field_num * args.embed_dim
        cin_layer_units = args.cin_layer_units
        self.cin = CIN(self.field_num, cin_layer_units)
        self.dnn = MLP(args.final_mlp_arch, input_dim, self.dropout)
        final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
        self.final_fc = nn.Linear(final_dim, args.output_dim)

        # 对于rerank任务，需要额外的CIN和DNN来处理包含user history的field
        if self.task == 'rerank' or self.task == 'rank':
            rerank_field_num = self.item_fnum + self.hist_fnum + self.augment_num
            rerank_input_dim = rerank_field_num * args.embed_dim
            self.cin_rerank = CIN(rerank_field_num, cin_layer_units)
            self.dnn_rerank = MLP(args.final_mlp_arch, rerank_input_dim, self.dropout)
            rerank_final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
            self.final_fc_rerank = nn.Linear(rerank_final_dim, args.output_dim)

    def forward(self, inp):
        if self.task == 'rerank' or self.task == 'rank':
            # Rerank任务：处理多个候选items
            item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
            batch_size = item_embedding.shape[0]
            max_list_len = item_embedding.shape[1]

            # 将user_behavior转换为与item_embedding兼容的表示
            user_repr = torch.mean(user_behavior, dim=1)  # (batch_size, hist_emb_dim)

            # 为每个候选item计算分数
            scores = []
            for i in range(max_list_len):
                item_emb = item_embedding[:, i, :]  # (batch_size, itm_emb_dim)

                # 将item和user嵌入分别reshape为field形式
                # item_emb: (batch_size, itm_emb_dim) -> reshape为 (batch_size, item_fnum, embed_dim)
                item_emb_field = item_emb.view(batch_size, self.item_fnum, self.embed_dim)
                # user_repr: (batch_size, hist_emb_dim) -> reshape为 (batch_size, hist_fnum, embed_dim)
                user_emb_field = user_repr.view(batch_size, self.hist_fnum, self.embed_dim)

                field_parts = [item_emb_field, user_emb_field]
                if self.augment_num and dens_vec is not None:
                    dens_emb = dens_vec[:, i, :].view(
                        batch_size, self.augment_num, self.embed_dim
                    )
                    field_parts.append(dens_emb)
                field_input = torch.cat(field_parts, dim=1)

                # 使用rerank任务专用的CIN和DNN
                final_vec = self.cin_rerank(field_input)
                dnn_vec = self.dnn_rerank(field_input.flatten(start_dim=1))
                final_vec = torch.cat([final_vec, dnn_vec], dim=1)
                logit = self.final_fc_rerank(final_vec)
                scores.append(logit)

            scores = torch.cat(scores, dim=1)  # (batch_size, max_list_len)
            scores = torch.sigmoid(scores)
            outputs = self.get_rerank_output(scores, labels)
            return outputs
        else:
            # CTR任务：原始逻辑
            inp, labels = self.get_filed_input(inp)
            final_vec = self.cin(inp)
            dnn_vec = self.dnn(inp.flatten(start_dim=1))
            final_vec = torch.cat([final_vec, dnn_vec], dim=1)
            logits = self.final_fc(final_vec)
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class AutoInt(BaseModel):
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
        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            field_inp, labels = self.get_filed_input(inp)
            attention_out = self.self_attention(field_inp)
            attention_out = torch.flatten(attention_out, start_dim=1)

            logits = self.attn_out(attention_out)
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class FiBiNet(BaseModel):
    def __init__(self, args, dataset):
        super(FiBiNet, self).__init__(args, dataset)
        self.senet_layer = SqueezeExtractionLayer(self.field_num, args.reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(self.embed_dim, self.field_num, args.bilinear_type)
        final_dim = self.field_num * (self.field_num - 1) * self.embed_dim
        self.dnn = MLP(args.final_mlp_arch, final_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            feat_embed, labels = self.get_filed_input(inp)
            senet_embed = self.senet_layer(feat_embed)
            bilinear_p = self.bilinear_layer(feat_embed)
            bilinear_q = self.bilinear_layer(senet_embed)
            comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)

            logits = self.fc_out(self.dnn(comb_out))
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class FiGNN(BaseModel):
    def __init__(self, args, dataset):
        super(FiGNN, self).__init__(args, dataset)
        self.fignn = FiGNNBlock(self.field_num, self.embed_dim, args.gnn_layer_num,
                                args.res_conn, args.reuse_graph_layer)
        self.fc = AttentionalPrediction(self.field_num, self.embed_dim)

    def forward(self, inp):
        if self.task == 'rerank' or self.task == 'rank':
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
        else:
            # CTR任务：原始逻辑
            feat_embed, labels = self.get_filed_input(inp)
            h_out = self.fignn(feat_embed)
            logits = self.fc(h_out)
            outputs = self.get_ctr_output(logits, labels)
            return outputs


class DLCM(BaseModel):
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


class PRM(BaseModel):
    def __init__(self, args, dataset):
        super(PRM, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.pos_embedding = torch.tensor(self.get_pos_embedding(self.max_list_len,
                                                                 self.module_inp_dim)).float().to(args.device)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def get_pos_embedding(self, max_len, d_emb):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        item_embed = processed_inp + self.pos_embedding

        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class SetRank(BaseModel):
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


class MIR(BaseModel):
    def __init__(self, args, dataset):
        super(MIR, self).__init__(args, dataset)
        self.intra_item_attn = nn.MultiheadAttention(self.itm_emb_dim, args.n_head, batch_first=True,
                                                     dropout=args.attn_dp)
        self.intra_hist_gru = nn.GRU(self.hist_emb_dim, self.hidden_size, dropout=self.rnn_dp,
                                     batch_first=True)
        self.i_fnum = self.item_fnum * 2
        self.h_fnum = self.hist_fnum + self.hidden_size // self.embed_dim

        self.set2list_attn = SLAttention(self.i_fnum, self.h_fnum, self.embed_dim,
                                         self.max_list_len, self.max_hist_len)
        self.mir_core_dim = (self.item_fnum * 3 + self.h_fnum * 2) * self.embed_dim
        self.mlp = MLP(
            args.final_mlp_arch,
            self.mir_core_dim + self.dens_vec_num,
            self.dropout,
        )
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        cross_item, _ = self.intra_item_attn(item_embedding, item_embedding, item_embedding)
        cross_hist, _ = self.intra_hist_gru(user_behavior)
        user_seq = torch.cat([user_behavior, cross_hist], dim=-1)
        hist_mean = torch.mean(user_seq, dim=1, keepdim=True)
        hist_mean = hist_mean.repeat([1, self.max_list_len, 1])
        cat_item = torch.cat([item_embedding, cross_item], dim=-1)

        v, q, _, _ = self.set2list_attn(cat_item, user_seq)
        mlp_inp = torch.cat([v, q, item_embedding, hist_mean], dim=-1)
        if self.augment_num and dens_vec is not None:
            mlp_inp = torch.cat([mlp_inp, dens_vec], dim=-1)
        mlp_out = self.mlp(mlp_inp)
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class GSF(BaseModel):
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


class EGRerank(BaseModel):
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


class LambdaRank(BaseModel):
    """
    LambdaRank-style Learning to Rank for Reranking
    Uses pairwise ranking loss with position-based weights
    """
    def __init__(self, args, dataset):
        super(LambdaRank, self).__init__(args, dataset)

        # Position-aware encoder
        self.pos_embedding = nn.Embedding(self.max_list_len, self.embed_dim)

        # Ranking score network
        self.rank_attn = nn.MultiheadAttention(
            self.module_inp_dim + self.embed_dim,
            args.n_head,
            batch_first=True,
            dropout=args.attn_dp
        )

        self.rank_mlp = MLP(args.final_mlp_arch, self.module_inp_dim + self.embed_dim, self.dropout)
        self.rank_fc = nn.Linear(args.final_mlp_arch[-1], 1)

    def compute_lambda_weights(self, scores, labels):
        """
        Compute LambdaRank weights based on NDCG
        """
        batch_size = scores.shape[0]

        # Compute pairwise differences
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)

        # Only consider pairs where labels differ
        valid_pairs = (label_diff != 0).float()

        # Position weights (top positions have higher weights)
        positions = torch.arange(1, self.max_list_len + 1, device=scores.device).float()
        position_weights = 1.0 / torch.log2(positions + 1)

        # Compute lambda weights
        lambda_weights = valid_pairs * position_weights.unsqueeze(0).unsqueeze(0)

        return lambda_weights

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        batch_size = item_embed.shape[0]

        # Add position embeddings
        positions = torch.arange(self.max_list_len, device=item_embed.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.pos_embedding(positions)
        item_embed_pos = torch.cat([item_embed, pos_emb], dim=-1)

        # Attention and scoring
        attn_out, _ = self.rank_attn(item_embed_pos, item_embed_pos, item_embed_pos)
        mlp_out = self.rank_mlp(attn_out)
        logits = self.rank_fc(mlp_out).squeeze(-1)
        scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

        outputs = self.get_rerank_output(scores, labels)
        return outputs


class RankFormer(BaseModel):
    """
    RankFormer (Buyl et al., SIGIR 2022) — 实现思路
    -----------------------------------------------
    使用 N 层 Transformer Encoder 对候选列表进行全局自注意力建模，
    同时引入一个列表级 [CLS] token 来预测 listwise rankability，
    并与 item-level sigmoid 评分融合，得到最终 rerank 分数。

    与 PRM 的区别：
      - 堆叠多层 (n_layers) Transformer，而不是单层 MHA；
      - 同时优化 list-level 和 item-level 目标（这里统一用 BCE 近似）；
      - 位置编码使用可学习的 nn.Embedding，更贴合短候选列表场景。
    """

    def __init__(self, args, dataset):
        super(RankFormer, self).__init__(args, dataset)
        n_layers = getattr(args, 'n_layers', 2)
        ff_dim = getattr(args, 'ff_dim', 128)

        self.pos_embedding = nn.Embedding(self.max_list_len + 1, self.module_inp_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.module_inp_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.module_inp_dim,
            nhead=args.n_head,
            dim_feedforward=ff_dim,
            dropout=args.attn_dp,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.item_mlp = MLP(args.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.item_fc = nn.Linear(args.final_mlp_arch[-1], 1)

        self.list_mlp = MLP(args.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.list_fc = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        bsz = item_embed.size(0)

        pos_ids = torch.arange(1, self.max_list_len + 1, device=item_embed.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0)
        x = item_embed + pos_emb

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)

        encoded = self.transformer(x)
        cls_repr, item_repr = encoded[:, 0, :], encoded[:, 1:, :]

        item_scores = torch.sigmoid(
            self.item_fc(self.item_mlp(item_repr))
        ).squeeze(-1)

        list_logit = torch.sigmoid(
            self.list_fc(self.list_mlp(cls_repr))
        )

        scores = item_scores * list_logit

        outputs = self.get_rerank_output(scores, labels)
        return outputs


class PEAR(BaseModel):
    """
    PEAR (Li et al., CIKM 2022) — 个性化 Rerank
    ---------------------------------------------
    核心: 将用户历史行为序列与候选列表通过两种注意力交互:
      1. list self-attention: 候选内部的相互影响；
      2. list→history cross-attention: 候选查询用户历史, 得到个性化上下文；
    再与原始候选拼接经过 MLP 打分。

    本实现为轻量版: 用一层 MHA 做 self-attn, 一层 MHA 做 cross-attn,
    历史行为先经 GRU 得到上下文化表示。
    """

    def __init__(self, args, dataset):
        super(PEAR, self).__init__(args, dataset)
        ff_dim = getattr(args, 'ff_dim', 128)

        self.hist_gru = nn.GRU(
            self.hist_emb_dim, self.hidden_size,
            dropout=self.rnn_dp, batch_first=True,
        )
        self.hist_proj = nn.Linear(self.hidden_size, self.module_inp_dim)

        self.self_attn = nn.MultiheadAttention(
            self.module_inp_dim, args.n_head,
            batch_first=True, dropout=args.attn_dp,
        )
        self.cross_attn = nn.MultiheadAttention(
            self.module_inp_dim, args.n_head,
            batch_first=True, dropout=args.attn_dp,
        )

        self.pos_embedding = nn.Embedding(self.max_list_len, self.module_inp_dim)

        self.fusion_mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 3, self.dropout)
        self.fusion_fc = nn.Linear(args.final_mlp_arch[-1], 1)

        self.ff = nn.Sequential(
            nn.Linear(self.module_inp_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ff_dim, self.module_inp_dim),
        )
        self.norm1 = nn.LayerNorm(self.module_inp_dim)
        self.norm2 = nn.LayerNorm(self.module_inp_dim)
        self.norm3 = nn.LayerNorm(self.module_inp_dim)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            item_embed = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            item_embed = item_embedding

        bsz = item_embed.size(0)
        pos_ids = torch.arange(self.max_list_len, device=item_embed.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        x = item_embed + pos_emb

        hist_ctx, _ = self.hist_gru(user_behavior)
        hist_ctx = self.hist_proj(hist_ctx)

        hist_mask = self.get_mask(hist_len, self.max_hist_len).squeeze(-1).bool()
        key_padding_mask = ~hist_mask

        self_out, _ = self.self_attn(x, x, x)
        x1 = self.norm1(x + self_out)

        cross_out, _ = self.cross_attn(x1, hist_ctx, hist_ctx,
                                        key_padding_mask=key_padding_mask)
        x2 = self.norm2(x1 + cross_out)
        x2 = self.norm3(x2 + self.ff(x2))

        fused = torch.cat([item_embed, x1, x2], dim=-1)
        logits = self.fusion_fc(self.fusion_mlp(fused)).squeeze(-1)
        scores = torch.sigmoid(logits)

        outputs = self.get_rerank_output(scores, labels)
        return outputs


class PIER(BaseModel):
    """
    PIER (Shi et al., KDD 2023) — Permutation-aware Rerank (lite 版本)
    ------------------------------------------------------------------
    原论文两阶段: FPSM (fine-grained permutation matching)
                + OCPM (omnidirectional cross-permutation matching).

    本实现保留其核心思想的轻量近似:
      - 局部置换建模: Bi-GRU, 让每个候选感知其相邻位置的 item;
      - 全局置换建模: Transformer self-attention, 捕获任意两两交互;
      - 双路信息通过 gate 融合, 提供 permutation-aware 表征;
      - 最终 MLP 打分.
    """

    def __init__(self, args, dataset):
        super(PIER, self).__init__(args, dataset)
        ff_dim = getattr(args, 'ff_dim', 128)

        self.pos_embedding = nn.Embedding(self.max_list_len, self.module_inp_dim)

        self.local_gru = nn.GRU(
            self.module_inp_dim, self.module_inp_dim // 2,
            dropout=self.rnn_dp, batch_first=True, bidirectional=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.module_inp_dim,
            nhead=args.n_head,
            dim_feedforward=ff_dim,
            dropout=args.attn_dp,
            batch_first=True,
            activation='gelu',
        )
        self.global_tf = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.gate = nn.Sequential(
            nn.Linear(self.module_inp_dim * 2, self.module_inp_dim),
            nn.Sigmoid(),
        )

        self.out_mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.out_fc = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        bsz = item_embed.size(0)

        pos_ids = torch.arange(self.max_list_len, device=item_embed.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        x = item_embed + pos_emb

        local_repr, _ = self.local_gru(x)
        global_repr = self.global_tf(x)

        gate = self.gate(torch.cat([local_repr, global_repr], dim=-1))
        perm_repr = gate * local_repr + (1 - gate) * global_repr

        fused = torch.cat([item_embed, perm_repr], dim=-1)
        logits = self.out_fc(self.out_mlp(fused)).squeeze(-1)
        scores = torch.sigmoid(logits)

        outputs = self.get_rerank_output(scores, labels)
        return outputs


# ============================================================================
# Rank Models (粗排阶段专用模型)
# 说明：Rank阶段复用CTR阶段的经典深度学习模型
# 这些模型都是2017-2019年提出的，在学术界和工业界广泛应用
# DeepFM (2017), xDeepFM (2018), DIN (2018), DIEN (2019)
# ============================================================================
#
# 注意：Rank阶段使用的DeepFM, xDeepFM, DIN, DIEN模型已经在前面定义
# 这里不需要重复定义，直接使用即可
# 在训练脚本中通过task='rerank'来区分CTR和Rank阶段


# ============================================================================
# Sequential Recommendation Baseline Models (序列推荐基线模型)
# 用于与知识增强模型对比的经典序列推荐方法
# ============================================================================

class GRU4Rec(BaseModel):
    """
    GRU4Rec: Session-based Recommendations with Recurrent Neural Networks (ICLR 2016)
    经典的RNN序列推荐模型，使用GRU建模用户行为序列
    """
    def __init__(self, args, dataset):
        super(GRU4Rec, self).__init__(args, dataset)

        # Item embedding
        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim, padding_idx=0)

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=args.hidden_size if hasattr(args, 'hidden_size') else 128,
            num_layers=args.num_layers if hasattr(args, 'num_layers') else 1,
            batch_first=True,
            dropout=self.dropout if args.num_layers > 1 else 0
        )

        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 128

        # Output layer
        if self.task == 'ctr':
            self.fc = nn.Linear(self.hidden_size, 1)
        elif self.task == 'rerank' or self.task == 'rank':
            self.fc = nn.Linear(self.hidden_size + self.embed_dim, 1)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)

            # Embed history
            hist_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, embed_dim]

            # GRU forward
            gru_out, _ = self.gru(hist_emb)  # [batch, max_hist_len, hidden_size]

            # Get last valid output for each sequence
            batch_size = hist_len.size(0)
            last_outputs = gru_out[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]

            # Dropout and predict
            last_outputs = self.dropout_layer(last_outputs)
            logits = self.fc(last_outputs).squeeze(-1)

            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed history
            hist_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, embed_dim]

            # GRU forward
            gru_out, _ = self.gru(hist_emb)  # [batch, max_hist_len, hidden_size]

            # Get last valid output
            last_outputs = gru_out[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]

            # Expand to match list length
            user_repr = last_outputs.unsqueeze(1).repeat(1, list_len, 1)  # [batch, list_len, hidden_size]

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, embed_dim]

            # Concatenate with item embeddings
            combined = torch.cat([user_repr, item_emb], dim=-1)  # [batch, list_len, hidden_size + embed_dim]

            # Dropout and predict
            combined = self.dropout_layer(combined)
            logits = self.fc(combined).squeeze(-1)  # [batch, list_len]
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

            outputs = self.get_rerank_output(scores, labels)
            return outputs


class Caser(BaseModel):
    """
    Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM 2018)
    使用CNN建模用户行为序列的推荐模型
    """
    def __init__(self, args, dataset):
        super(Caser, self).__init__(args, dataset)

        # Parameters
        self.L = args.max_hist_len  # Sequence length
        self.d = self.embed_dim  # Embedding dimension
        self.nh = args.num_h_filters if hasattr(args, 'num_h_filters') else 16  # Horizontal filters
        self.nv = args.num_v_filters if hasattr(args, 'num_v_filters') else 4   # Vertical filters

        # Item embedding
        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim, padding_idx=0)

        # Horizontal convolutional layers (different heights)
        self.conv_h = nn.ModuleList([
            nn.Conv2d(1, self.nh, (h, self.d))
            for h in [2, 3, 4, 5]  # Different filter heights
        ])

        # Vertical convolutional layer
        self.conv_v = nn.Conv2d(1, self.nv, (self.L, 1))

        # Fully connected layer
        fc_input_dim = self.nh * len(self.conv_h) + self.nv * self.d

        if self.task == 'ctr':
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim, 1)
            )
        elif self.task == 'rerank' or self.task == 'rank':
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim + self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embed_dim, 1)
            )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)

            # Embed history
            hist_emb = self.item_embedding(hist_iid)  # [batch, L, d]
            hist_emb = hist_emb.unsqueeze(1)  # [batch, 1, L, d]

            # Horizontal convolutions
            h_outs = []
            for conv in self.conv_h:
                h_out = torch.relu(conv(hist_emb)).squeeze(3)  # [batch, nh, L']
                h_out = torch.max_pool1d(h_out, h_out.size(2)).squeeze(2)  # [batch, nh]
                h_outs.append(h_out)
            h_out = torch.cat(h_outs, dim=1)  # [batch, nh * 4]

            # Vertical convolution
            v_out = torch.relu(self.conv_v(hist_emb)).squeeze(2)  # [batch, nv, d]
            v_out = v_out.view(v_out.size(0), -1)  # [batch, nv * d]

            # Concatenate and predict
            combined = torch.cat([h_out, v_out], dim=1)  # [batch, fc_input_dim]
            combined = self.dropout_layer(combined)
            logits = self.fc(combined).squeeze(-1)

            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed history
            hist_emb = self.item_embedding(hist_iid)  # [batch, L, d]
            hist_emb = hist_emb.unsqueeze(1)  # [batch, 1, L, d]

            # Horizontal convolutions
            h_outs = []
            for conv in self.conv_h:
                h_out = torch.relu(conv(hist_emb)).squeeze(3)  # [batch, nh, L']
                h_out = torch.max_pool1d(h_out, h_out.size(2)).squeeze(2)  # [batch, nh]
                h_outs.append(h_out)
            h_out = torch.cat(h_outs, dim=1)  # [batch, nh * 4]

            # Vertical convolution
            v_out = torch.relu(self.conv_v(hist_emb)).squeeze(2)  # [batch, nv, d]
            v_out = v_out.view(v_out.size(0), -1)  # [batch, nv * d]

            # User representation
            user_repr = torch.cat([h_out, v_out], dim=1)  # [batch, fc_input_dim]
            user_repr = user_repr.unsqueeze(1).repeat(1, list_len, 1)  # [batch, list_len, fc_input_dim]

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, embed_dim]

            # Concatenate with item embeddings
            combined = torch.cat([user_repr, item_emb], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.fc(combined).squeeze(-1)  # [batch, list_len]
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

            outputs = self.get_rerank_output(scores, labels)
            return outputs


class SASRec(BaseModel):
    """
    SASRec: Self-Attentive Sequential Recommendation (ICDM 2018)
    使用Self-Attention机制建模用户行为序列
    """
    def __init__(self, args, dataset):
        super(SASRec, self).__init__(args, dataset)

        # Parameters
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 64
        self.num_blocks = args.num_blocks if hasattr(args, 'num_blocks') else 2
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 2

        # Item embedding
        self.item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_hist_len, self.hidden_size)

        # Self-attention blocks
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
            self.hidden_size,
                self.num_heads,
                dropout=self.dropout,
            batch_first=True
        )
            for _ in range(self.num_blocks)
        ])

        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Dropout(self.dropout)
            )
            for _ in range(self.num_blocks)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range(self.num_blocks * 2)
        ])

        # Output layer
        if self.task == 'ctr':
            self.fc = nn.Linear(self.hidden_size, 1)
        elif self.task == 'rerank' or self.task == 'rank':
            self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)
            batch_size = hist_iid.size(0)

            # Embed items and add positional encoding
            seq_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, hidden_size]
            positions = torch.arange(self.max_hist_len, device=hist_iid.device).unsqueeze(0).repeat(batch_size, 1)
            pos_emb = self.position_embedding(positions)
            seq_emb = seq_emb + pos_emb

            # Create attention mask (causal mask + padding mask)
            _, key_padding_mask = self._create_attention_mask(hist_iid, hist_len)

            # Self-attention blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                # Self-attention
                attn_out, _ = self.attention_layers[i](hidden, hidden, hidden, key_padding_mask=key_padding_mask)
                hidden = self.layer_norms[i * 2](hidden + attn_out)

                # Feed-forward
                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]

            # Predict
            logits = self.fc(last_outputs).squeeze(-1)
            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed items and add positional encoding
            seq_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, hidden_size]
            positions = torch.arange(self.max_hist_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            pos_emb = self.position_embedding(positions)
            seq_emb = seq_emb + pos_emb

            # Create attention mask
            _, key_padding_mask = self._create_attention_mask(hist_iid, hist_len)

            # Self-attention blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                attn_out, _ = self.attention_layers[i](hidden, hidden, hidden, key_padding_mask=key_padding_mask)
                hidden = self.layer_norms[i * 2](hidden + attn_out)

                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]
            user_repr = last_outputs.unsqueeze(1).repeat(1, list_len, 1)

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, hidden_size]

            # Concatenate and predict
            combined = torch.cat([user_repr, item_emb], dim=-1)
            logits = self.fc(combined).squeeze(-1)
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

        outputs = self.get_rerank_output(scores, labels)
        return outputs

    def _create_attention_mask(self, seq, lengths):
        """Create causal + padding mask for self-attention"""
        batch_size, seq_len = seq.size()

        # Causal mask (lower triangular) - [seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Padding mask - [batch, seq_len]
        key_padding_mask = torch.arange(seq_len, device=seq.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # For PyTorch MultiheadAttention:
        # - attn_mask: [seq_len, seq_len] or [batch*num_heads, seq_len, seq_len]
        # - key_padding_mask: [batch, seq_len]
        # We'll use key_padding_mask and return both
        return None, key_padding_mask  # Return None for attn_mask, use key_padding_mask


class BERT4Rec(BaseModel):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer (CIKM 2019)
    使用双向Transformer建模用户行为序列
    """
    def __init__(self, args, dataset):
        super(BERT4Rec, self).__init__(args, dataset)

        # Parameters
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 64
        self.num_blocks = args.num_blocks if hasattr(args, 'num_blocks') else 2
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 2
        self.mask_prob = args.mask_prob if hasattr(args, 'mask_prob') else 0.2

        # Item embedding (including mask token)
        self.item_embedding = nn.Embedding(self.item_num + 2, self.hidden_size, padding_idx=0)  # +2 for padding and mask
        self.position_embedding = nn.Embedding(self.max_hist_len, self.hidden_size)

        # Bidirectional self-attention blocks
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                self.hidden_size,
                self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
            for _ in range(self.num_blocks)
        ])

        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Dropout(self.dropout)
            )
            for _ in range(self.num_blocks)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range(self.num_blocks * 2)
        ])

        # Output layer
        if self.task == 'ctr':
            self.fc = nn.Linear(self.hidden_size, 1)
        elif self.task == 'rerank' or self.task == 'rank':
            self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)
            batch_size = hist_iid.size(0)

            # Embed items and add positional encoding
            seq_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, hidden_size]
            positions = torch.arange(self.max_hist_len, device=hist_iid.device).unsqueeze(0).repeat(batch_size, 1)
            pos_emb = self.position_embedding(positions)
            seq_emb = seq_emb + pos_emb

            # Create padding mask (bidirectional, only mask padding)
            key_padding_mask = torch.arange(self.max_hist_len, device=hist_iid.device).unsqueeze(0) >= hist_len.unsqueeze(1)

            # Bidirectional self-attention blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                # Self-attention (bidirectional)
                attn_out, _ = self.attention_layers[i](hidden, hidden, hidden, key_padding_mask=key_padding_mask)
                hidden = self.layer_norms[i * 2](hidden + attn_out)

                # Feed-forward
                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]

            # Predict
            logits = self.fc(last_outputs).squeeze(-1)
            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed items and add positional encoding
            seq_emb = self.item_embedding(hist_iid)  # [batch, max_hist_len, hidden_size]
            positions = torch.arange(self.max_hist_len, device=device).unsqueeze(0).repeat(batch_size, 1)
            pos_emb = self.position_embedding(positions)
            seq_emb = seq_emb + pos_emb

            # Create padding mask
            key_padding_mask = torch.arange(self.max_hist_len, device=device).unsqueeze(0) >= hist_len.unsqueeze(1)

            # Bidirectional self-attention blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                attn_out, _ = self.attention_layers[i](hidden, hidden, hidden, key_padding_mask=key_padding_mask)
                hidden = self.layer_norms[i * 2](hidden + attn_out)

                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]  # [batch, hidden_size]
            user_repr = last_outputs.unsqueeze(1).repeat(1, list_len, 1)

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, hidden_size]

            # Concatenate and predict
            combined = torch.cat([user_repr, item_emb], dim=-1)
            logits = self.fc(combined).squeeze(-1)
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

            outputs = self.get_rerank_output(scores, labels)
            return outputs


class NARM(BaseModel):
    """
    NARM: Neural Attentive Session-based Recommendation (WWW 2017)
    结合GRU和注意力机制的会话推荐模型
    """
    def __init__(self, args, dataset):
        super(NARM, self).__init__(args, dataset)

        # Parameters
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 128
        self.num_layers = args.num_layers if hasattr(args, 'num_layers') else 1

        # Item embedding
        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim, padding_idx=0)

        # GRU for encoding
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # Attention layers
        self.A_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.A_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)

        # Output layers
        if self.task == 'ctr':
            self.ct_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.fc = nn.Linear(self.hidden_size, 1)
        elif self.task == 'rerank' or self.task == 'rank':
            self.ct_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.fc = nn.Linear(self.hidden_size + self.embed_dim, 1)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)
            batch_size = hist_iid.size(0)

            # Embed history
            hist_emb = self.item_embedding(hist_iid)

            # GRU encoding
            gru_out, hidden = self.gru(hist_emb)

            # Get last valid hidden state (global encoding)
            ht = gru_out[torch.arange(batch_size), hist_len - 1]

            # Compute attention weights
            mask = torch.arange(self.max_hist_len, device=hist_iid.device).unsqueeze(0) < hist_len.unsqueeze(1)

            # Attention computation
            q1 = self.A_1(gru_out)
            q2 = self.A_2(ht).unsqueeze(1)
            q = torch.sigmoid(q1 + q2)

            alpha = self.v_t(q).squeeze(-1)
            alpha = alpha.masked_fill(~mask, -1e9)
            alpha = torch.softmax(alpha, dim=-1)

            # Weighted sum (local encoding)
            c_t = torch.sum(alpha.unsqueeze(-1) * gru_out, dim=1)

            # Combine global and local
            combined = torch.cat([ht, c_t], dim=-1)
            combined = self.ct_mlp(combined)

            # Predict
            logits = self.fc(combined).squeeze(-1)
            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed history
            hist_emb = self.item_embedding(hist_iid)

            # GRU encoding
            gru_out, hidden = self.gru(hist_emb)

            # Get last valid hidden state
            ht = gru_out[torch.arange(batch_size), hist_len - 1]

            # Compute attention
            mask = torch.arange(self.max_hist_len, device=device).unsqueeze(0) < hist_len.unsqueeze(1)

            q1 = self.A_1(gru_out)
            q2 = self.A_2(ht).unsqueeze(1)
            q = torch.sigmoid(q1 + q2)

            alpha = self.v_t(q).squeeze(-1)
            alpha = alpha.masked_fill(~mask, -1e9)
            alpha = torch.softmax(alpha, dim=-1)

            c_t = torch.sum(alpha.unsqueeze(-1) * gru_out, dim=1)

            # Combine global and local
            combined = torch.cat([ht, c_t], dim=-1)
            user_repr = self.ct_mlp(combined)

            # Expand to match list length
            user_repr = user_repr.unsqueeze(1).repeat(1, list_len, 1)

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, embed_dim]

            # Concatenate with item embeddings
            combined = torch.cat([user_repr, item_emb], dim=-1)
            combined = self.dropout_layer(combined)
            logits = self.fc(combined).squeeze(-1)
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

            outputs = self.get_rerank_output(scores, labels)
            return outputs


class FMLPRec(BaseModel):
    """
    FMLP-Rec: Filter-enhanced MLP for Sequential Recommendation (WWW 2022)
    使用纯MLP和Fourier变换的序列推荐模型
    """
    def __init__(self, args, dataset):
        super(FMLPRec, self).__init__(args, dataset)

        # Parameters
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 64
        self.num_blocks = args.num_blocks if hasattr(args, 'num_blocks') else 2

        # Item embedding
        self.item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)

        # Learnable Fourier transform (frequency filter)
        self.complex_weight = nn.Parameter(
            torch.randn(self.num_blocks, self.max_hist_len // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02
        )

        # Feed-forward layers
        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Dropout(self.dropout)
            )
            for _ in range(self.num_blocks)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range(self.num_blocks * 2)
        ])

        # Output layer
        if self.task == 'ctr':
            self.fc = nn.Linear(self.hidden_size, 1)
        elif self.task == 'rerank' or self.task == 'rank':
            self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, inp):
        if self.task == 'ctr':
            item_id, hist_iid, hist_len, dens_vec, orig_dens_vec, labels = self.process_ctr_input(inp)
            batch_size = hist_iid.size(0)

            # Embed items
            seq_emb = self.item_embedding(hist_iid)

            # Create mask
            mask = torch.arange(self.max_hist_len, device=hist_iid.device).unsqueeze(0) < hist_len.unsqueeze(1)
            seq_emb = seq_emb * mask.unsqueeze(-1).float()

            # FMLP blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                # Filter layer (in frequency domain)
                x = hidden
                x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

                # Apply learnable frequency filter
                weight = torch.view_as_complex(self.complex_weight[i])
                x_fft = x_fft * weight.unsqueeze(0)

                x = torch.fft.irfft(x_fft, n=self.max_hist_len, dim=1, norm='ortho')
                hidden = self.layer_norms[i * 2](hidden + x)

                # Feed-forward
                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]

            # Predict
            logits = self.fc(last_outputs).squeeze(-1)
            outputs = self.get_ctr_output(logits, labels)
            return outputs

        elif self.task == 'rerank' or self.task == 'rank':
            device = next(self.parameters()).device
            # 直接从输入获取原始ID
            hist_iid = inp['hist_iid_seq'].to(device)  # [batch, max_hist_len]
            hist_len = inp['hist_seq_len'].to(device)  # [batch]
            iid_list = inp['iid_list'].to(device)  # [batch, list_len]
            labels = inp['lb_list'].to(device)  # [batch, list_len]

            batch_size, list_len = iid_list.size()

            # Embed items
            seq_emb = self.item_embedding(hist_iid)

            # Create mask
            mask = torch.arange(self.max_hist_len, device=device).unsqueeze(0) < hist_len.unsqueeze(1)
            seq_emb = seq_emb * mask.unsqueeze(-1).float()

            # FMLP blocks
            hidden = seq_emb
            for i in range(self.num_blocks):
                # Filter layer
                x = hidden
                x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

                weight = torch.view_as_complex(self.complex_weight[i])
                x_fft = x_fft * weight.unsqueeze(0)

                x = torch.fft.irfft(x_fft, n=self.max_hist_len, dim=1, norm='ortho')
                hidden = self.layer_norms[i * 2](hidden + x)

                ff_out = self.forward_layers[i](hidden)
                hidden = self.layer_norms[i * 2 + 1](hidden + ff_out)

            # Get last valid output
            last_outputs = hidden[torch.arange(batch_size), hist_len - 1]
            user_repr = last_outputs.unsqueeze(1).repeat(1, list_len, 1)

            # Embed candidate items
            item_emb = self.item_embedding(iid_list)  # [batch, list_len, hidden_size]

            # Concatenate and predict
            combined = torch.cat([user_repr, item_emb], dim=-1)
            logits = self.fc(combined).squeeze(-1)
            scores = torch.sigmoid(logits)  # Apply sigmoid for BCELoss

            outputs = self.get_rerank_output(scores, labels)
            return outputs


class BSARec(BaseModel):
    """
    BSARec: Beyond Self-Attention for Sequential Recommendation (AAAI 2024)
    通过傅里叶变换结合归纳偏置与自注意力，融合低频和高频信息，
    解决 Transformer 序列推荐中的过平滑问题。
    论文: https://arxiv.org/abs/2312.10325
    """
    def __init__(self, args, dataset):
        super(BSARec, self).__init__(args, dataset)
        self.hidden_size = getattr(args, 'hidden_size', 128)
        self.num_blocks = getattr(args, 'num_blocks', 2)
        self.num_heads = getattr(args, 'n_head', 2)
        self.alpha = getattr(args, 'bsarec_alpha', 0.5)

        self.seq_item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)

        self.bsa_blocks = nn.ModuleList([
            _BSABlock(self.hidden_size, self.num_heads, self.max_hist_len, self.alpha, self.dropout)
            for _ in range(self.num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        if self.task == 'ctr':
            self.fc = nn.Linear(self.hidden_size, 1)
        else:
            # user_repr: hidden_size, item_emb: hidden_size (seq_item_embedding)
            out_dim = self.hidden_size * 2
            if self.augment_num and self.dens_vec_num > 0:
                out_dim += self.dens_vec_num
            self.fc = nn.Linear(out_dim, 1)

    def _encode_sequence(self, hist_iid, hist_len):
        device = hist_iid.device
        batch_size = hist_iid.size(0)
        seq_emb = self.seq_item_embedding(hist_iid)
        mask = (hist_iid != 0)
        hidden = self.layer_norm(seq_emb)
        for block in self.bsa_blocks:
            hidden = block(hidden, mask)
        last_idx = (hist_len - 1).clamp(min=0)
        return hidden[torch.arange(batch_size, device=device), last_idx]

    def forward(self, inp):
        device = next(self.parameters()).device

        if self.task == 'ctr':
            hist_iid = inp['hist_iid_seq'].to(device)
            hist_len = inp['hist_seq_len'].to(device)
            iid = inp['iid'].to(device)
            labels = inp['lb'].to(device)
            user_repr = self._encode_sequence(hist_iid, hist_len)
            item_emb = self.seq_item_embedding(iid)
            logits = self.fc(user_repr + item_emb).squeeze(-1)
            return self.get_ctr_output(logits, labels)

        else:  # rank / rerank
            hist_iid = inp['hist_iid_seq'].to(device)
            hist_len = inp['hist_seq_len'].to(device)
            iid_list = inp['iid_list'].to(device)
            labels = inp['lb_list'].to(device)
            batch_size, list_len = iid_list.size()

            user_repr = self._encode_sequence(hist_iid, hist_len)
            item_emb = self.seq_item_embedding(iid_list)
            user_exp = user_repr.unsqueeze(1).expand(-1, list_len, -1)

            if self.augment_num and self.dens_vec_num > 0:
                _, _, _, dens_vec, _, _ = self.process_input(inp)
                if dens_vec is not None:
                    if len(dens_vec.shape) == 2:
                        dens_vec = dens_vec.unsqueeze(1).expand(-1, list_len, -1)
                    combined = torch.cat([user_exp, item_emb, dens_vec], dim=-1)
                else:
                    combined = torch.cat([user_exp, item_emb], dim=-1)
            else:
                combined = torch.cat([user_exp, item_emb], dim=-1)

            logits = self.fc(combined).squeeze(-1)
            scores = torch.sigmoid(logits)
            return self.get_rerank_output(scores, labels)


class _BSABlock(nn.Module):
    """BSARec 基础块：FFT 低通滤波 + 自注意力高通，通过 alpha 融合。"""
    def __init__(self, hidden_size, num_heads, max_seq_len, alpha, dropout):
        super().__init__()
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.complex_weight = nn.Parameter(
            torch.randn(max_seq_len // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        xf = torch.fft.rfft(x, n=self.max_seq_len, dim=1, norm='ortho')
        w = torch.view_as_complex(self.complex_weight)
        xf = xf * w.unsqueeze(0)
        x_low = torch.fft.irfft(xf, n=self.max_seq_len, dim=1, norm='ortho')[:, :seq_len, :]

        attn_key_mask = (~mask) if mask is not None else None
        x_high, _ = self.attn(x, x, x, key_padding_mask=attn_key_mask)

        x = self.norm1(x + self.alpha * x_low + (1 - self.alpha) * x_high)
        x = self.norm2(x + self.ff(x))
        return x
