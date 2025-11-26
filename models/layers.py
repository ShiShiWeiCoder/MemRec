from itertools import combinations, product

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class Dice(nn.Module):
    """
    activation function DICE in DIN
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, fc_dims, input_dim, dropout):
        super(MLP, self).__init__()
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    """
    Mixture of Export
    """
    def __init__(self, moe_arch, inp_dim, dropout):
        super(MoE, self).__init__()
        export_num, export_arch = moe_arch
        self.export_num = export_num
        self.gate_net = nn.Linear(inp_dim, export_num)
        self.export_net = nn.ModuleList([MLP(export_arch, inp_dim, dropout) for _ in range(export_num)])

    def forward(self, x):
        gate = self.gate_net(x).view(-1, self.export_num)  # (bs, export_num)
        gate = nn.functional.softmax(gate, dim=-1).unsqueeze(dim=1) # (bs, 1, export_num)
        experts = [net(x) for net in self.export_net]
        experts = torch.stack(experts, dim=1)  # (bs, expert_num, emb)
        out = torch.matmul(gate, experts).squeeze(dim=1)
        return out


class HEA(nn.Module):
    """
    hybrid-expert adaptor with long-short interest attention fusion
    """
    def __init__(self, ple_arch, inp_dim, dropout, enable_ls_attention=False, num_attn_heads=4):
        super(HEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = ple_arch
        self.enable_ls_attention = enable_ls_attention
        
        self.share_expt_net = nn.ModuleList([MLP(expt_arch, inp_dim, dropout) for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.ModuleList([nn.ModuleList([MLP(expt_arch, inp_dim, dropout)
                                                           for _ in range(spcf_expt_num)]) for _ in range(task_num)])
        self.gate_net = nn.ModuleList([nn.Linear(inp_dim, share_expt_num + spcf_expt_num)
                                   for _ in range(task_num)])
        
        # 添加长短兴趣多头注意力融合
        if self.enable_ls_attention:
            self.ls_attention_fusion = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=num_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            print(f"HEA: 启用长短兴趣多头注意力融合 (头数: {num_attn_heads})")

    def forward(self, x_list, ls_ratios=None):
        """
        Args:
            x_list: 输入向量列表 [用户历史增强向量, 物品增强向量, 分析增强向量等]
            ls_ratios: 长短兴趣比例信息 [(short_ratio, long_ratio), ...]
        """
        # 确保输入向量数量与专家网络数量匹配
        actual_tower_num = len(x_list)
        expected_tower_num = len(self.spcf_expt_net)
        
        if actual_tower_num != expected_tower_num:
            # 截断或填充输入向量列表以匹配专家网络数量
            if actual_tower_num > expected_tower_num:
                x_list = x_list[:expected_tower_num]
                if ls_ratios:
                    ls_ratios = ls_ratios[:expected_tower_num]
            else:
                # 用最后一个向量填充
                last_x = x_list[-1]
                x_list.extend([last_x] * (expected_tower_num - actual_tower_num))
                if ls_ratios:
                    last_ratio = ls_ratios[-1] if ls_ratios[-1] is not None else None
                    ls_ratios.extend([last_ratio] * (expected_tower_num - actual_tower_num))
        
        # 如果启用长短兴趣注意力融合且提供了比例信息
        if self.enable_ls_attention and ls_ratios is not None:
            fused_x_list = []
            for i, (x, ls_ratio) in enumerate(zip(x_list, ls_ratios)):
                if ls_ratio is not None:
                    short_ratio, long_ratio = ls_ratio
                    
                    # 处理批次维度的比例张量
                    if short_ratio.dim() > 0:
                        # 取第一个样本的比例作为代表（假设批次内比例相似）
                        short_ratio_scalar = short_ratio[0].item()
                        long_ratio_scalar = long_ratio[0].item()
                    else:
                        short_ratio_scalar = short_ratio.item()
                        long_ratio_scalar = long_ratio.item()
                    
                    # 将输入向量分解为短期和长期部分（基于比例）
                    dim = x.size(-1)
                    short_dim = int(dim * short_ratio_scalar)
                    long_dim = dim - short_dim
                    
                    if short_dim > 0 and long_dim > 0:
                        short_repr = x[..., :short_dim]
                        long_repr = x[..., short_dim:]
                        
                        # 填充到原始输入维度以便注意力计算
                        # 确保填充后的维度与原始输入x的维度一致
                        target_dim = dim  # 使用原始输入的维度
                        
                        if short_dim < target_dim:
                            padding_size = target_dim - short_dim
                            padding = torch.zeros(short_repr.shape[:-1] + (padding_size,), 
                                                device=short_repr.device, dtype=short_repr.dtype)
                            short_repr = torch.cat([short_repr, padding], dim=-1)
                        
                        if long_dim < target_dim:
                            padding_size = target_dim - long_dim
                            padding = torch.zeros(long_repr.shape[:-1] + (padding_size,), 
                                                device=long_repr.device, dtype=long_repr.dtype)
                            long_repr = torch.cat([long_repr, padding], dim=-1)
                        
                        # 使用多头注意力融合长短期表示
                        fused_repr, _ = self.ls_attention_fusion(
                            query=short_repr.unsqueeze(1),  # 短期作为query
                            key=long_repr.unsqueeze(1),     # 长期作为key
                            value=x.unsqueeze(1)            # 原始向量作为value
                        )
                        fused_x_list.append(fused_repr.squeeze(1))
                    else:
                        fused_x_list.append(x)
                else:
                    fused_x_list.append(x)
            x_list = fused_x_list
        
        # 原有的专家网络处理
        gates = [net(x) for net, x in zip(self.gate_net, x_list)]
        gates = torch.stack(gates, dim=1)  # (bs, tower_num, expert_num), export_num = share_expt_num + spcf_expt_num
        gates = nn.functional.softmax(gates, dim=-1).unsqueeze(dim=2)  # (bs, tower_num, 1, expert_num)
        cat_x = torch.stack(x_list, dim=1)  # (bs, tower_num, inp_dim)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = torch.stack(share_experts, dim=2)  # (bs, tower_num, share_expt_num, E)
        
        # 修复特定专家网络的处理逻辑
        spcf_experts = []
        for i, (nets, x) in enumerate(zip(self.spcf_expt_net, x_list)):
            tower_experts = [net(x) for net in nets]
            tower_experts = torch.stack(tower_experts, dim=1)  # (bs, spcf_expt_num, E)
            spcf_experts.append(tower_experts)
        spcf_experts = torch.stack(spcf_experts, dim=1)  # (bs, tower_num, spcf_expt_num, E)
        
        experts = torch.cat([share_experts, spcf_experts], dim=2)  # (bs, tower_num, expert_num, E)
        export_mix = torch.matmul(gates, experts).squeeze(dim=2)  # (bs, tower_num, E)
        # print('export mix', export_mix.shape, 'tower num', self.tower_num)
        export_mix = torch.split(export_mix, dim=1, split_size_or_sections=1)
        out = [x.squeeze(dim=1) for x in export_mix]
        return out


class ConvertNet(nn.Module):
    """
    convert from semantic space to recommendation space
    """
    def __init__(self, args, inp_dim, dropout, conv_type):
        super(ConvertNet, self).__init__()
        self.type = conv_type
        self.dropout = dropout
        self.device = args.device
        
        # 保存参数用于延迟初始化
        self.transformer_heads = getattr(args, 'transformer_heads', 8)
        self.transformer_layers = getattr(args, 'transformer_layers', 2)
        self.feedforward_dim = getattr(args, 'transformer_ff_dim', None)
        self.activation = getattr(args, 'transformer_activation', 'gelu')
        
        if conv_type in ['TransformerHEA', 'HEA']:
            # 对于HEA和TransformerHEA，使用PLE架构
            # 使用完整的convert_arch作为专家架构
            ple_arch = [args.export_num, args.specific_export_num, args.convert_arch, args.augment_num]
            enable_ls_attention = getattr(args, 'enable_ls_attention', False)
            num_attn_heads = getattr(args, 'ls_attn_heads', 4)
            
            if conv_type == 'TransformerHEA':
                self.sub_module = TransformerHEA(
                    ple_arch, inp_dim, dropout, 
                    enable_ls_attention=enable_ls_attention,
                    num_attn_heads=num_attn_heads,
                    transformer_heads=self.transformer_heads,
                    transformer_layers=self.transformer_layers,
                    feedforward_dim=self.feedforward_dim,
                    activation=self.activation,
                    expected_output_dim_per_task=args.convert_arch[-1]
                )
            else:
                self.sub_module = HEA(ple_arch, inp_dim, dropout, enable_ls_attention, num_attn_heads)
        elif conv_type in ['MultilevelMemoryHEA', 'MultilevelMemoryMoE']:
            if conv_type == 'MultilevelMemoryHEA':
                ple_arch = [args.export_num, args.specific_export_num, args.convert_arch, args.augment_num]
                sensory_heads = getattr(args, 'sensory_attn_heads', 2)
                working_heads = getattr(args, 'working_attn_heads', 2)
                longterm_heads = getattr(args, 'longterm_attn_heads', 2)
                enable_attention = getattr(args, 'enable_memory_attention', True)
                memory_fusion_type = getattr(args, 'memory_fusion_type', 'hierarchical')
                
                self.sub_module = MultilevelMemoryHEA(
                    ple_arch, inp_dim, dropout,
                    sensory_attn_heads=sensory_heads,
                    working_attn_heads=working_heads,
                    longterm_attn_heads=longterm_heads,
                    enable_multilevel_attention=enable_attention,
                    memory_fusion_type=memory_fusion_type
                )
            else:
                moe_arch = [args.export_num, args.convert_arch]
                sensory_heads = getattr(args, 'sensory_attn_heads', 2)
                working_heads = getattr(args, 'working_attn_heads', 2)
                longterm_heads = getattr(args, 'longterm_attn_heads', 2)
                enable_attention = getattr(args, 'enable_memory_attention', True)
                
                self.sub_module = MultilevelMemoryMoE(
                    moe_arch, inp_dim, dropout,
                    sensory_attn_heads=sensory_heads,
                    working_attn_heads=working_heads,
                    longterm_attn_heads=longterm_heads,
                    enable_multilevel_attention=enable_attention
                )
        elif conv_type == 'TransformerMoE':
            # TransformerMoE需要延迟初始化
            self.sub_module = None
            self.moe_arch = [args.export_num, args.convert_arch[0]]
            # 添加输出投影层，将TransformerMoE的输出投影到期望的维度
            self.output_projection = None
            self.expected_output_dim = args.convert_arch[-1] * args.augment_num
        elif conv_type == 'MoE':
            # 原始MoE实现
            moe_arch = [args.export_num, args.convert_arch[0]]
            self.sub_module = MoE(moe_arch, inp_dim, dropout)
        else:
            # MLP实现
            self.sub_module = MLP(args.convert_arch, inp_dim, dropout)
        
    def forward(self, x_list, ls_ratios=None, multilevel_memory_data=None):
        if self.type in ['TransformerHEA', 'HEA']:
            out = self.sub_module(x_list, ls_ratios)
            if isinstance(out, list):
                out = torch.cat(out, dim=-1)
        elif self.type in ['MultilevelMemoryHEA', 'MultilevelMemoryMoE']:
            if self.type == 'MultilevelMemoryHEA':
                out = self.sub_module(x_list, multilevel_memory_data)
                if isinstance(out, list):
                    out = torch.cat(out, dim=-1)
            else:
                concatenated_input = torch.cat(x_list, dim=-1)
                out = self.sub_module(concatenated_input, multilevel_memory_data)
        elif self.type == 'TransformerMoE':
            # TransformerMoE期望单个输入，所以先拼接x_list
            concatenated_input = torch.cat(x_list, dim=-1)
            
            # 延迟初始化TransformerMoE以适应实际输入维度
            if self.sub_module is None:
                actual_inp_dim = concatenated_input.shape[-1]
                self.sub_module = TransformerMoE(
                    moe_arch=self.moe_arch,
                    inp_dim=actual_inp_dim,
                    dropout=self.dropout,
                    num_heads=self.transformer_heads,
                    num_layers=self.transformer_layers,
                    feedforward_dim=self.feedforward_dim,
                    activation=self.activation
                ).to(self.device)
                
                # 初始化输出投影层
                self.output_projection = nn.Linear(actual_inp_dim, self.expected_output_dim).to(self.device)
            
            out, aux_loss = self.sub_module(concatenated_input)
            # 投影到期望的输出维度
            out = self.output_projection(out)
            
            # 保存aux_loss供后续使用
            if hasattr(self, '_aux_loss'):
                self._aux_loss = aux_loss
            else:
                self._aux_loss = aux_loss
        else:
            # 原始实现：MoE或MLP
            out = [self.sub_module(x) for x in x_list]
            out = torch.cat(out, dim=-1)
        return out


class AttentionPoolingLayer(nn.Module):
    """
      attention pooling in DIN
    """

    def __init__(self, embedding_dim, dropout, fc_dims=[32, 16]):
        super(AttentionPoolingLayer, self).__init__()
        fc_layers = []
        input_dim = embedding_dim * 4
        # fc layer
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior, mask=None):
        """
          :param query_ad:embedding of target item   -> (bs, dim)
          :param user_behavior:embedding of user behaviors     ->  (bs, seq_len, dim)
          :param mask:mask on user behaviors  ->  (bs,seq_len, 1)
          :return output:user interest (bs, dim)
        """
        query = query.unsqueeze(1)
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior,
                                queries * user_behavior], dim=-1)
        attns = self.fc(attn_input)
        if mask is not None:
            attns = attns.mul(mask)
        out = user_behavior.mul(attns)
        output = out.sum(dim=1)
        return output, attns


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** :dimension of input feature
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: cross net
        - **mode**: "v1"  or "v2" ,DCNv1 or DCNv2
    """

    def __init__(self, inp_dim, layer_num=2, mode='v1'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.mode = mode
        if self.mode == 'v1': # DCN
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList([
                nn.Parameter(nn.init.xavier_normal_(torch.empty(inp_dim, 1))) for _ in range(self.layer_num)])
        elif self.mode == 'v2': # DCNv2
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([
                nn.Parameter(nn.init.xavier_normal_(torch.empty(inp_dim, inp_dim))) for _ in range(self.layer_num)])
        else:  # error
            raise ValueError("mode in CrossNet should be 'v1' or 'v2'")

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(inp_dim, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.mode == 'v1':
            # x0 * (xl.T * w )+ b + xl
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.mode == 'v2':
            # x0 * (Wxl + bl) + xl
                dot_ = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                raise ValueError("mode in CrossNet should be 'v1' or 'v2'")

            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units):
        super(CIN, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, X_0):
        pooling_outputs = []
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        return concate_vec


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, layer_norm=False, align_to="input"):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            if align_to == "output":
                self.W_res = nn.Linear(input_dim, self.output_dim, bias=False)
            elif align_to == "input":
                self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query

        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            if self.align_to == "output":  # AutoInt style
                residual = self.W_res(residual)
            elif self.align_to == "input":  # Transformer stype
                output = self.W_res(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class SqueezeExtractionLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio):
        super(SqueezeExtractionLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(nn.Module):
    def __init__(self, embed_size, num_fields, bilinear_type):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(embed_size, embed_size, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for _ in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for _, _ in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class GraphLayer(nn.Module):
    def __init__(self, num_fields, embed_size):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embed_size, embed_size))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embed_size, embed_size))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embed_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)  # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNNBlock(nn.Module):
    def __init__(self, num_fields, embed_size, gnn_layer_num, res_conn, reuse_graph_layer):
        super(FiGNNBlock, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embed_size
        self.gnn_layers = gnn_layer_num
        self.use_residual = res_conn
        self.reuse_graph_layer = reuse_graph_layer
        if self.reuse_graph_layer:
            self.gnn = GraphLayer(self.num_fields, self.embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(self.num_fields, self.embedding_dim) for _ in range(self.gnn_layers)])
        self.gru = nn.GRUCell(embed_size, embed_size)
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embed_size * 2, 1, bias=False)

    def build_graph_with_attention(self, feat_embed):
        src_emb = feat_embed[:, self.src_nodes, :]
        dst_emb = feat_embed[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        try:
            device = feat_embed.get_device()
            mask = torch.eye(self.num_fields).to(device)
        except RuntimeError:
            mask = torch.eye(self.num_fields)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = nn.functional.softmax(alpha, dim=-1)  # batch x field x field without self-loops
        return graph

    def forward(self, feat_embed):
        g = self.build_graph_with_attention(feat_embed)
        h = feat_embed
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feat_embed
        return h


class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embed_size):
        super(AttentionalPrediction, self).__init__()
        self.linear1 = nn.Linear(embed_size, 1, bias=False)
        self.linear2 = nn.Sequential(nn.Linear(num_fields * embed_size, num_fields, bias=False),
                                     nn.Sigmoid())

    def forward(self, h):
        score = self.linear1(h).squeeze(-1)
        weight = self.linear2(h.flatten(start_dim=1))
        logits = (weight * score).sum(dim=1).unsqueeze(-1)
        return logits


class InterestExtractor(nn.Module):
    """
    Interest extractor in DIEN
    """
    def __init__(self, input_size, hidden_size):
        super(InterestExtractor, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, keys, keys_length):
        """
        keys:        [btz, seq_len, hdsz]
        keys_length: [btz, 1]
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        if keys_length.shape[0] == 0:
            return torch.zeros(btz, hdsz, device=keys.device)

        masked_keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)  # 去除全为0序列的样本
        packed_keys = pack_padded_sequence(masked_keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0, total_length=seq_len)

        return interests


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)


class InterestEvolving(nn.Module):
    """
    Interest evolving in DIEN
    """

    def __init__(self, input_size, gru_type='GRU', dropout=0):
        super(InterestEvolving, self).__init__()
        assert gru_type in {'GRU', 'AIGRU', 'AGRU', 'AUGRU'}, f"gru_type: {gru_type} is not supported"
        self.gru_type = gru_type

        if gru_type == 'GRU':
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size, gru_type=gru_type)

        self.attention = AttentionPoolingLayer(embedding_dim=input_size, dropout=dropout)


    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        query:       [btz, 1, hdsz]
        keys:        [btz, seq_len ,hdsz]
        keys_length: [btz, 1]
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        zero_outputs = torch.zeros(btz, hdsz, device=query.device)
        if keys_length.shape[0] == 0:
            return zero_outputs

        query = torch.masked_select(query, smp_mask.view(-1, 1)).view(-1, hdsz)
        keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)  # 去除全为0序列的样本

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=seq_len)
            outputs, _ = self.attention(query, interests, mask)  # [btz1, hdsz]

        elif self.gru_type == 'AIGRU':
            _, att_scores = self.attention(query, keys, mask)  # [btz1, 1, seq_len]
            interests = keys * att_scores # [btz1, seq_len, hdsz]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [btz1, hdsz]

        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            _, att_scores = self.attention(query, keys, mask) # [b, T]
            att_scores = att_scores.squeeze(1)
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=seq_len)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length)  # [b, H]

        # [b, H] -> [B, H]
        zero_outputs[smp_mask.squeeze(1)] = outputs
        return zero_outputs


class Phi_function(nn.Module):
    """
    phi function on
    """

    def __init__(self, input_size, hidden_size, dropout=0):
        super(Phi_function, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.dp1 = torch.nn.Dropout(dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)

    def forward(self, seq_state, final_state):
        bn1 = self.bn1(final_state)
        fc1 = self.fc1(bn1)
        dp1 = self.dp1(self.tanh(fc1))
        bn2 = self.bn2(seq_state.transpose(1, 2)).transpose(1, 2)
        fc2 = self.fc2(torch.unsqueeze(dp1, dim=1) * bn2)
        score = torch.softmax(fc2, dim=-1)
        seq_len = score.shape[1]
        score = score[:, :, 0].view([-1, seq_len])
        return score


class SLAttention(nn.Module):
    """
    SLAttention for MIR
    v for item, q for hist
    """
    def __init__(self, v_fnum, q_fnum, emb_dim, v_len, q_len, fi=True, ii=True):
        super(SLAttention, self).__init__()
        self.v_fnum = v_fnum
        self.q_fnum = q_fnum
        self.emb_dim = emb_dim
        self.v_dim = v_fnum * emb_dim
        self.q_dim = q_fnum * emb_dim
        self.v_len = v_len
        self.q_len = q_len
        self.fi = fi
        self.ii = ii
        if fi:
            self.w_b_fi = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.emb_dim, self.emb_dim)))
            self.fi_conv = nn.Conv2d(1, 1, kernel_size=(self.q_fnum, self.v_fnum), stride=(self.q_fnum, self.v_fnum))
        if ii:
            self.w_b_ii = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.q_dim, self.v_dim)))

        self.w_v = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.v_dim, self.v_len)))
        self.w_q = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.q_dim, self.v_len)))

    def forward(self, V, Q):
        batch_size = V.shape[0]
        if self.ii:
            w_b_ii = self.w_b_ii.repeat([batch_size, 1, 1])
            V_trans = V.permute(0, 2, 1)
            C1 = torch.matmul(Q, torch.matmul(w_b_ii, V_trans))
            C = torch.tan(C1)
        if self.fi:
            V_s = V.view(-1, self.v_len * self.v_fnum, self.emb_dim)
            Q_s = Q.view(-1, self.q_len * self.q_fnum, self.emb_dim)
            w_b_fi = self.w_b_fi.repeat([batch_size, 1, 1])
            V_s_trans = V_s.permute(0, 2, 1)
            C2 = torch.matmul(Q_s, torch.matmul(w_b_fi, V_s_trans)).unsqueeze(1)

            C2 = self.fi_conv(C2)
            C2 = C2.view(-1, self.q_len, self.v_len)
            if self.ii:
                C = torch.tanh(C1 + C2)
            else:
                C = torch.tanh(C2)

        hv_1 = torch.matmul(V.view(-1, self.v_dim), self.w_v).view(-1, self.v_len, self.v_len)
        hq_1 = torch.matmul(Q.view(-1, self.q_dim), self.w_q).view(-1, self.q_len, self.v_len)
        hq_1 = hq_1.permute(0, 2, 1)
        h_v = torch.tanh(hv_1 + torch.matmul(hq_1, C))
        h_q = torch.tanh(hq_1 + torch.matmul(hv_1, C.permute(0, 2, 1)))
        a_v = torch.softmax(h_v, dim=-1)
        a_q = torch.softmax(h_q, dim=-1)
        v = torch.matmul(a_v, V)
        q = torch.matmul(a_q, Q)
        return v, q, a_v, a_q


class TransformerExpert(nn.Module):
    """
    基于Transformer的专家网络
    """
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, dropout=0.1, 
                 feedforward_dim=None, use_layer_norm=True, activation='gelu'):
        super(TransformerExpert, self).__init__()
        
        # 确保hidden_dim能被num_heads整除
        if hidden_dim % num_heads != 0:
            # 自动调整hidden_dim到最接近的能被num_heads整除的值
            adjusted_hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
            print(f"⚠️ 警告: hidden_dim ({hidden_dim}) 不能被 num_heads ({num_heads}) 整除")
            print(f"   自动调整 hidden_dim: {hidden_dim} -> {adjusted_hidden_dim}")
            hidden_dim = adjusted_hidden_dim
        
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim) if use_layer_norm else None
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # 残差连接
        self.use_residual = (input_dim == hidden_dim)
        if not self.use_residual:
            self.residual_projection = nn.Linear(input_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) 或 (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, input_dim)
        """
        # 处理输入维度
        if x.dim() == 2:
            # (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x_input = x.unsqueeze(1)
            squeeze_output = True
        else:
            x_input = x
            squeeze_output = False
        
        residual = x_input
        
        # 输入投影
        projected = self.input_projection(x_input)  # (batch_size, seq_len, hidden_dim)
        
        # Transformer编码
        # 注意：nn.TransformerEncoder期望 (seq_len, batch_size, hidden_dim) 或 batch_first=True时为 (batch_size, seq_len, hidden_dim)
        encoded = self.transformer_encoder(projected)  # (batch_size, seq_len, hidden_dim)
        
        # 输出投影
        output = self.output_projection(encoded)  # (batch_size, seq_len, input_dim)
        
        # 残差连接
        if self.use_residual:
            output = output + residual
        else:
            output = output + self.residual_projection(residual)
        
        # Dropout
        output = self.dropout(output)
        
        # 如果输入是2D，则压缩输出
        if squeeze_output:
            output = output.squeeze(1)  # (batch_size, input_dim)
        
        return output


class TransformerMoE(nn.Module):
    """
    基于Transformer的混合专家网络
    """
    def __init__(self, moe_arch, inp_dim, dropout, num_heads=8, num_layers=2, 
                 feedforward_dim=None, activation='gelu'):
        super(TransformerMoE, self).__init__()
        expert_num, expert_hidden_dim = moe_arch
        self.expert_num = expert_num
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(inp_dim, expert_num * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_num * 2, expert_num)
        )
        
        # Transformer专家网络
        self.expert_net = nn.ModuleList([
            TransformerExpert(
                input_dim=inp_dim,
                hidden_dim=expert_hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                feedforward_dim=feedforward_dim,
                activation=activation
            ) for _ in range(expert_num)
        ])
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            output: (batch_size, input_dim)
            aux_loss: 负载均衡损失
        """
        batch_size = x.size(0)
        
        # 门控网络
        gate_logits = self.gate_net(x)  # (batch_size, expert_num)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch_size, expert_num)
        
        # 专家网络计算
        expert_outputs = []
        for expert in self.expert_net:
            expert_output = expert(x)  # (batch_size, input_dim)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, expert_num, input_dim)
        
        # 加权融合
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch_size, expert_num, 1)
        output = torch.sum(gate_weights_expanded * expert_outputs, dim=1)  # (batch_size, input_dim)
        
        # 计算负载均衡损失
        aux_loss = self._compute_load_balance_loss(gate_weights)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(self, gate_weights):
        """
        计算负载均衡损失，鼓励专家网络的均匀使用
        """
        # 计算每个专家的平均权重
        expert_usage = torch.mean(gate_weights, dim=0)  # (expert_num,)
        
        # 理想情况下每个专家的使用率应该是 1/expert_num
        ideal_usage = 1.0 / self.expert_num
        
        # 计算方差作为负载均衡损失
        load_balance_loss = torch.var(expert_usage) * self.load_balance_weight
        
        return load_balance_loss


class TransformerHEA(nn.Module):
    """
    基于Transformer的混合专家适配器 (Hybrid Expert Adaptor)
    """
    def __init__(self, ple_arch, inp_dim, dropout, enable_ls_attention=False, 
                 num_attn_heads=4, transformer_heads=8, transformer_layers=2,
                 feedforward_dim=None, activation='gelu', expected_output_dim_per_task=None):
        super(TransformerHEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_hidden_dim, task_num = ple_arch
        self.enable_ls_attention = enable_ls_attention
        self.task_num = task_num
        
        # 共享Transformer专家网络
        self.share_expt_net = nn.ModuleList([
            TransformerExpert(
                input_dim=inp_dim,
                hidden_dim=expt_hidden_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                feedforward_dim=feedforward_dim,
                activation=activation
            ) for _ in range(share_expt_num)
        ])
        
        # 任务特定Transformer专家网络
        self.spcf_expt_net = nn.ModuleList([
            nn.ModuleList([
                TransformerExpert(
                    input_dim=inp_dim,
                    hidden_dim=expt_hidden_dim,
                    num_heads=transformer_heads,
                    num_layers=transformer_layers,
                    dropout=dropout,
                    feedforward_dim=feedforward_dim,
                    activation=activation
                ) for _ in range(spcf_expt_num)
            ]) for _ in range(task_num)
        ])
        
        # 门控网络 - 使用更复杂的门控机制
        gate_hidden_dim = max(64, (share_expt_num + spcf_expt_num) * 2)
        self.gate_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, share_expt_num + spcf_expt_num),
                nn.Softmax(dim=-1)
            ) for _ in range(task_num)
        ])
        
        # 长短兴趣多头注意力融合
        if self.enable_ls_attention:
            self.ls_attention_fusion = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=num_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            print(f"TransformerHEA: 启用长短兴趣多头注意力融合 (头数: {num_attn_heads})")
        
        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(inp_dim * task_num, inp_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inp_dim * 2, inp_dim)
        )
        
        # 添加输出投影层，将每个任务的输出投影到期望的维度
        # 这确保了与原始HEA的输出维度兼容性
        # 修复：使用convert_arch的最后一层维度而不是专家隐藏维度
        # 从ple_arch推断convert_arch的最后一层维度
        # ple_arch = [share_expt_num, spcf_expt_num, expt_hidden_dim, task_num]
        # 在ConvertNet中，期望的输出维度是convert_arch[-1] * task_num
        # 所以每个任务的输出维度应该是convert_arch[-1]
        # 我们需要从外部传入这个信息，暂时使用一个合理的默认值
        if expected_output_dim_per_task is None:
            expected_output_dim_per_task = 32  # 默认使用32，这通常是convert_arch[-1]的值
        self.task_output_projections = nn.ModuleList([
            nn.Linear(inp_dim, expected_output_dim_per_task) for _ in range(task_num)
        ])

    def forward(self, x_list, ls_ratios=None):
        """
        Args:
            x_list: 输入向量列表 [用户历史增强向量, 物品增强向量, 分析增强向量等]
            ls_ratios: 长短兴趣比例信息 [(short_ratio, long_ratio), ...]
        Returns:
            output: 融合后的输出向量
        """
        # 确保输入向量数量与任务数量匹配
        actual_tower_num = len(x_list)
        expected_tower_num = self.task_num
        
        if actual_tower_num != expected_tower_num:
            if actual_tower_num > expected_tower_num:
                x_list = x_list[:expected_tower_num]
                if ls_ratios:
                    ls_ratios = ls_ratios[:expected_tower_num]
            else:
                last_x = x_list[-1]
                x_list.extend([last_x] * (expected_tower_num - actual_tower_num))
                if ls_ratios:
                    last_ratio = ls_ratios[-1] if ls_ratios else None
                    ls_ratios.extend([last_ratio] * (expected_tower_num - actual_tower_num))
        
        # 长短兴趣注意力融合（如果启用）
        if self.enable_ls_attention and ls_ratios is not None:
            fused_x_list = []
            for i, (x, ls_ratio) in enumerate(zip(x_list, ls_ratios)):
                if ls_ratio is not None:
                    short_ratio, long_ratio = ls_ratio
                    
                    # 处理批次维度的比例张量
                    if short_ratio.dim() > 0:
                        short_ratio_scalar = short_ratio[0].item()
                        long_ratio_scalar = long_ratio[0].item()
                    else:
                        short_ratio_scalar = short_ratio.item()
                        long_ratio_scalar = long_ratio.item()
                    
                    # 分解为短期和长期部分
                    dim = x.size(-1)
                    short_dim = int(dim * short_ratio_scalar)
                    long_dim = dim - short_dim
                    
                    if short_dim > 0 and long_dim > 0:
                        short_repr = x[..., :short_dim]
                        long_repr = x[..., short_dim:]
                        
                        # 填充到原始维度
                        if short_dim < dim:
                            padding_size = dim - short_dim
                            padding = torch.zeros(short_repr.shape[:-1] + (padding_size,), 
                                                device=short_repr.device, dtype=short_repr.dtype)
                            short_repr = torch.cat([short_repr, padding], dim=-1)
                        
                        if long_dim < dim:
                            padding_size = dim - long_dim
                            padding = torch.zeros(long_repr.shape[:-1] + (padding_size,), 
                                                device=long_repr.device, dtype=long_repr.dtype)
                            long_repr = torch.cat([long_repr, padding], dim=-1)
                        
                        # 多头注意力融合
                        fused_repr, _ = self.ls_attention_fusion(
                            query=short_repr.unsqueeze(1),
                            key=long_repr.unsqueeze(1),
                            value=x.unsqueeze(1)
                        )
                        fused_x_list.append(fused_repr.squeeze(1))
                    else:
                        fused_x_list.append(x)
                else:
                    fused_x_list.append(x)
            x_list = fused_x_list
        
        # 门控网络计算
        gates = []
        for i, (gate_net, x) in enumerate(zip(self.gate_net, x_list)):
            gate_weights = gate_net(x)  # (batch_size, expert_num)
            gates.append(gate_weights)
        
        # 共享专家网络计算
        share_expert_outputs = []
        for expert in self.share_expt_net:
            # 对所有任务的输入计算共享专家输出
            expert_outputs_per_task = []
            for x in x_list:
                expert_output = expert(x)  # (batch_size, input_dim)
                expert_outputs_per_task.append(expert_output)
            share_expert_outputs.append(torch.stack(expert_outputs_per_task, dim=1))  # (batch_size, task_num, input_dim)
        
        # 任务特定专家网络计算
        spcf_expert_outputs = []
        for task_id, (task_experts, x) in enumerate(zip(self.spcf_expt_net, x_list)):
            task_expert_outputs = []
            for expert in task_experts:
                expert_output = expert(x)  # (batch_size, input_dim)
                task_expert_outputs.append(expert_output)
            spcf_expert_outputs.append(torch.stack(task_expert_outputs, dim=1))  # (batch_size, spcf_expert_num, input_dim)
        
        # 专家输出融合
        task_outputs = []
        for task_id in range(self.task_num):
            # 获取该任务的门控权重
            gate_weights = gates[task_id]  # (batch_size, total_expert_num)
            
            # 分离共享专家和特定专家的权重
            share_expert_num = len(self.share_expt_net)
            share_weights = gate_weights[:, :share_expert_num]  # (batch_size, share_expert_num)
            spcf_weights = gate_weights[:, share_expert_num:]   # (batch_size, spcf_expert_num)
            
            # 融合共享专家输出
            share_output = torch.zeros_like(x_list[task_id])  # (batch_size, input_dim)
            for expert_id, expert_outputs in enumerate(share_expert_outputs):
                expert_weight = share_weights[:, expert_id:expert_id+1]  # (batch_size, 1)
                share_output += expert_weight * expert_outputs[:, task_id, :]  # (batch_size, input_dim)
            
            # 融合特定专家输出
            spcf_output = torch.zeros_like(x_list[task_id])  # (batch_size, input_dim)
            task_spcf_outputs = spcf_expert_outputs[task_id]  # (batch_size, spcf_expert_num, input_dim)
            for expert_id in range(task_spcf_outputs.size(1)):
                expert_weight = spcf_weights[:, expert_id:expert_id+1]  # (batch_size, 1)
                spcf_output += expert_weight * task_spcf_outputs[:, expert_id, :]  # (batch_size, input_dim)
            
            # 任务输出 = 共享专家输出 + 特定专家输出
            task_output = share_output + spcf_output
            task_outputs.append(task_output)
        
        # 最终融合所有任务输出
        concatenated_output = torch.cat(task_outputs, dim=-1)  # (batch_size, input_dim * task_num)
        final_output = self.output_fusion(concatenated_output)  # (batch_size, input_dim)
        
        # 投影每个任务输出到期望的维度
        projected_task_outputs = []
        for task_id, task_output in enumerate(task_outputs):
            projected_output = self.task_output_projections[task_id](task_output)
            projected_task_outputs.append(projected_output)
        
        return projected_task_outputs


class MultilevelMemoryHEA(nn.Module):
    def __init__(self, ple_arch, inp_dim, dropout, 
                 sensory_attn_heads=2, working_attn_heads=2, longterm_attn_heads=2,
                 enable_multilevel_attention=True, memory_fusion_type='hierarchical'):
        super(MultilevelMemoryHEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = ple_arch
        self.enable_multilevel_attention = enable_multilevel_attention
        self.memory_fusion_type = memory_fusion_type
        self.task_num = task_num
        
        # 🧠 记忆引导的注意力增强机制
        if self.enable_multilevel_attention:
            # 📌 线性变换层：从VN_analysis中提取各级记忆向量
            # 论文公式: V_m^memory = W_m · VN_analysis
            self.W_sensory = nn.Linear(inp_dim, inp_dim)   # 提取感觉记忆
            self.W_working = nn.Linear(inp_dim, inp_dim)   # 提取工作记忆
            self.W_longterm = nn.Linear(inp_dim, inp_dim)  # 提取长期记忆
            
            # 🔥 用户向量增强注意力（使用感觉记忆注意力模块）
            self.sensory_attention = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=sensory_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            
            # ⚡ 课程向量增强注意力（使用工作记忆注意力模块）
            self.working_attention = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=working_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            
            print(f"MultilevelMemoryHEA: 启用记忆引导的注意力增强机制（论文版本）")
            print(f"  📌 从VN_analysis提取三级记忆并自适应融合")
            print(f"  🔥 用户向量增强头数: {sensory_attn_heads}")
            print(f"  ⚡ 课程向量增强头数: {working_attn_heads}")
        
        # 🚀 MLP专家网络 (共享 + 任务特定)
        # 共享MLP专家 - 所有任务共用
        self.share_expt_net = nn.ModuleList([
            MLP(expt_arch, inp_dim, dropout) for _ in range(share_expt_num)
        ])
        
        # 任务特定MLP专家 - 每个任务独有
        self.spcf_expt_net = nn.ModuleList([
            nn.ModuleList([
                MLP(expt_arch, inp_dim, dropout) for _ in range(spcf_expt_num)
            ]) for _ in range(task_num)
        ])
        
        # 🎛️ 门控网络 - 控制专家权重分配
        gate_hidden_dim = max(64, (share_expt_num + spcf_expt_num) * 2)
        self.gate_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, share_expt_num + spcf_expt_num),
                nn.Softmax(dim=-1)
            ) for _ in range(task_num)
        ])
        
        # 📊 记忆权重调节层 - 基于记忆比例动态调整
        self.memory_weight_adaptor = nn.Sequential(
            nn.Linear(3, inp_dim // 4),  # 3个记忆比例输入
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inp_dim // 4, 3),  # 输出3个记忆权重
            nn.Softmax(dim=-1)
        )
        
        print(f"MultilevelMemoryHEA: MLP专家网络配置")
        print(f"  共享专家数: {share_expt_num}, 特定专家数: {spcf_expt_num}")
        print(f"  任务数: {task_num}, 专家架构: {expt_arch}")
    
    def forward(self, x_list, multilevel_memory_data=None):
        """
        前向传播（优化版本）
        
        设计思路：
        - x_list = [VNuser, VNcourse, VNanalysis]
        - 从VNanalysis提取并融合三级记忆得到V_memory
        - 用V_memory分别增强VNuser和VNcourse
        - 🎯 只将增强后的两个向量输入专家网络（避免信息冗余）
        
        Args:
            x_list: 输入向量列表 [VNuser, VNcourse, VNanalysis]
            multilevel_memory_data: 多级记忆数据字典（包含记忆比例）
        Returns:
            task_outputs: 每个任务的输出列表（2个任务：用户增强向量、课程增强向量）
        """
        # 1️⃣ 记忆引导的注意力增强
        if multilevel_memory_data is not None and self.enable_multilevel_attention and len(x_list) >= 3:
            VNuser = x_list[0]
            VNcourse = x_list[1]
            VNanalysis = x_list[2]
            
            # 步骤1: 从VNanalysis提取三级记忆
            # 论文公式: V_m^memory = W_m · VN_analysis
            V_sensory = self.W_sensory(VNanalysis)
            V_working = self.W_working(VNanalysis)
            V_longterm = self.W_longterm(VNanalysis)
            
            device = VNuser.device
            sensory_ratio = multilevel_memory_data.get('sensory_memory_ratio', torch.tensor(0.33, device=device))
            working_ratio = multilevel_memory_data.get('working_memory_ratio', torch.tensor(0.33, device=device))
            longterm_ratio = multilevel_memory_data.get('longterm_memory_ratio', torch.tensor(0.34, device=device))
            
            # 论文公式: ω = Softmax(MLP([ratio_sensory, ratio_working, ratio_longterm]))
            memory_ratio_tensor = torch.stack([sensory_ratio, working_ratio, longterm_ratio], dim=-1)
            if memory_ratio_tensor.dim() == 1:
                memory_ratio_tensor = memory_ratio_tensor.unsqueeze(0).expand(VNuser.size(0), -1)
            
            adaptive_weights = self.memory_weight_adaptor(memory_ratio_tensor)  # (batch_size, 3)
            adaptive_weights = adaptive_weights.unsqueeze(-1)  # (batch_size, 3, 1)
            
            # 论文公式: V_memory = Σ ω_m · V_m^memory
            memory_stack = torch.stack([V_sensory, V_working, V_longterm], dim=1)  # (batch_size, 3, inp_dim)
            V_memory = torch.sum(memory_stack * adaptive_weights, dim=1)  # (batch_size, inp_dim)
            
            # 步骤3: 用V_memory分别增强VNuser和VNcourse
            # 论文公式: V_user^enh = MultiHeadAttention(VN_user, V_memory, V_memory)
            V_user_enh, _ = self.sensory_attention(
                query=VNuser.unsqueeze(1),
                key=V_memory.unsqueeze(1),
                value=V_memory.unsqueeze(1)
            )
            V_user_enh = V_user_enh.squeeze(1)
            
            # 论文公式: V_course^enh = MultiHeadAttention(VN_course, V_memory, V_memory)
            V_course_enh, _ = self.working_attention(
                query=VNcourse.unsqueeze(1),
                key=V_memory.unsqueeze(1),
                value=V_memory.unsqueeze(1)
            )
            V_course_enh = V_course_enh.squeeze(1)
            
            # 🎯 优化：只保留增强后的两个向量，VNanalysis的信息已融入前两个
            # 这样可以减少计算成本，同时避免信息冗余
            x_list = [V_user_enh, V_course_enh]
        elif len(x_list) > 2:
            # 如果多级记忆数据不存在但输入仍有3个向量，只使用前两个
            x_list = x_list[:2]
        
        # 2️⃣ MLP专家网络处理
        # 门控网络计算
        gates = []
        for i, (gate_net, x) in enumerate(zip(self.gate_net, x_list)):
            gate_weights = gate_net(x)  # (batch_size, expert_num)
            gates.append(gate_weights)
        
        # 共享MLP专家网络计算
        share_expert_outputs = []
        for expert in self.share_expt_net:
            # 对所有任务的输入计算共享专家输出
            expert_outputs_per_task = []
            for x in x_list:
                expert_output = expert(x)  # (batch_size, expert_arch[-1])
                expert_outputs_per_task.append(expert_output)
            share_expert_outputs.append(torch.stack(expert_outputs_per_task, dim=1))  # (batch_size, task_num, expert_arch[-1])
        
        # 任务特定MLP专家网络计算
        spcf_expert_outputs = []
        for task_id, (task_experts, x) in enumerate(zip(self.spcf_expt_net, x_list)):
            task_expert_outputs = []
            for expert in task_experts:
                expert_output = expert(x)  # (batch_size, expert_arch[-1])
                task_expert_outputs.append(expert_output)
            spcf_expert_outputs.append(torch.stack(task_expert_outputs, dim=1))  # (batch_size, spcf_expert_num, expert_arch[-1])
        
        # 3️⃣ 专家输出融合
        task_outputs = []
        for task_id in range(self.task_num):
            # 获取该任务的门控权重
            gate_weights = gates[task_id]  # (batch_size, total_expert_num)
            
            # 分离共享专家和特定专家的权重
            share_expert_num = len(self.share_expt_net)
            share_weights = gate_weights[:, :share_expert_num]  # (batch_size, share_expert_num)
            spcf_weights = gate_weights[:, share_expert_num:]   # (batch_size, spcf_expert_num)
            
            # 融合共享专家输出
            # 获取专家网络的输出维度 (expt_arch的最后一个值)
            expert_output_dim = share_expert_outputs[0].size(-1) if share_expert_outputs else 64  # 默认64
            share_output = torch.zeros(x_list[task_id].size(0), expert_output_dim, device=x_list[task_id].device)
            for expert_id, expert_outputs in enumerate(share_expert_outputs):
                expert_weight = share_weights[:, expert_id:expert_id+1]  # (batch_size, 1)
                share_output += expert_weight * expert_outputs[:, task_id, :]  # (batch_size, expert_arch[-1])
            
            # 融合特定专家输出  
            spcf_output = torch.zeros_like(share_output)
            task_spcf_outputs = spcf_expert_outputs[task_id]  # (batch_size, spcf_expert_num, expert_arch[-1])
            for expert_id in range(task_spcf_outputs.size(1)):
                expert_weight = spcf_weights[:, expert_id:expert_id+1]  # (batch_size, 1)
                spcf_output += expert_weight * task_spcf_outputs[:, expert_id, :]  # (batch_size, expert_arch[-1])
            
            # 任务输出 = 共享专家输出 + 特定专家输出
            task_output = share_output + spcf_output
            task_outputs.append(task_output)
        
        return task_outputs


class MultilevelMemoryMoE(nn.Module):
    def __init__(self, moe_arch, inp_dim, dropout, 
                 sensory_attn_heads=2, working_attn_heads=2, longterm_attn_heads=2,
                 enable_multilevel_attention=True):
        super(MultilevelMemoryMoE, self).__init__()
        expert_num, expert_arch = moe_arch
        self.expert_num = expert_num
        self.enable_multilevel_attention = enable_multilevel_attention
        
        if self.enable_multilevel_attention:
            # 🔥 感觉记忆注意力
            self.sensory_attention = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=sensory_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            
            # ⚡ 工作记忆注意力
            self.working_attention = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=working_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            
            # 🏗️ 长期记忆注意力
            self.longterm_attention = MultiHeadAttention(
                input_dim=inp_dim,
                num_heads=longterm_attn_heads,
                dropout_rate=dropout,
                use_residual=True,
                layer_norm=True,
                align_to="input"
            )
            
        self.gate_net = nn.Sequential(
            nn.Linear(inp_dim, expert_num * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_num * 2, expert_num),
            nn.Softmax(dim=-1)
        )
        
        # 🚀 MLP专家网络
        self.expert_net = nn.ModuleList([
            MLP(expert_arch, inp_dim, dropout) for _ in range(expert_num)
        ])
        
        print(f"MultilevelMemoryMoE: MLP专家网络配置")
        print(f"  专家数: {expert_num}, 专家架构: {expert_arch}")
    
    def multilevel_memory_fusion(self, x, memory_vecs):
        """简化版多级记忆融合"""
        if not self.enable_multilevel_attention:
            return x
        
        # 三级记忆注意力处理 (简化版)
        sensory_vec = memory_vecs.get('sensory_memory_vec', torch.zeros_like(x))
        working_vec = memory_vecs.get('working_memory_vec', torch.zeros_like(x))
        longterm_vec = memory_vecs.get('longterm_memory_vec', torch.zeros_like(x))
        
        # 注意力融合
        x_expanded = x.unsqueeze(1)
        sensory_fused, _ = self.sensory_attention(x_expanded, sensory_vec.unsqueeze(1), sensory_vec.unsqueeze(1))
        working_fused, _ = self.working_attention(x_expanded, working_vec.unsqueeze(1), working_vec.unsqueeze(1))
        longterm_fused, _ = self.longterm_attention(x_expanded, longterm_vec.unsqueeze(1), longterm_vec.unsqueeze(1))
        
        # 简单加权融合
        fused_x = x + 0.3 * sensory_fused.squeeze(1) + 0.4 * working_fused.squeeze(1) + 0.3 * longterm_fused.squeeze(1)
        return fused_x
    
    def forward(self, x, multilevel_memory_data=None):
        if multilevel_memory_data is not None and self.enable_multilevel_attention:
            memory_vecs = {
                'sensory_memory_vec': multilevel_memory_data.get('sensory_memory_vec', torch.zeros_like(x)),
                'working_memory_vec': multilevel_memory_data.get('working_memory_vec', torch.zeros_like(x)),
                'longterm_memory_vec': multilevel_memory_data.get('longterm_memory_vec', torch.zeros_like(x))
            }
            x = self.multilevel_memory_fusion(x, memory_vecs)
        
        # 2️⃣ 门控网络
        gate_weights = self.gate_net(x)  # (batch_size, expert_num)
        
        # 3️⃣ MLP专家网络计算
        expert_outputs = []
        for expert in self.expert_net:
            expert_output = expert(x)  # (batch_size, expert_arch[-1])
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, expert_num, expert_arch[-1])
        
        # 4️⃣ 加权融合
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch_size, expert_num, 1)
        output = torch.sum(gate_weights_expanded * expert_outputs, dim=1)  # (batch_size, expert_arch[-1])
        
        return output
