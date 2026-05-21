import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, fc_dims, input_dim, dropout=0.0):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in fc_dims:
            layers.extend([nn.Linear(last_dim, dim), nn.ReLU(), nn.Dropout(dropout)])
            last_dim = dim
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, x):
        return self.net(x)


class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, query, user_behavior, mask=None):
        query = query.unsqueeze(1).expand_as(user_behavior)
        scores = self.proj(torch.cat([query, user_behavior, query - user_behavior, query * user_behavior], dim=-1)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * user_behavior).sum(dim=1)


class CrossNet(nn.Module):
    def __init__(self, inp_dim, layer_num=2):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(inp_dim, 1) * 0.01) for _ in range(layer_num)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(inp_dim)) for _ in range(layer_num)])

    def forward(self, inputs):
        x0 = inputs
        x = inputs
        for weight, bias in zip(self.weights, self.biases):
            x = x0 * torch.matmul(x, weight) + bias + x
        return x


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units):
        super().__init__()
        self.num_fields = num_fields
        self.layer_units = cin_layer_units

    def forward(self, x0):
        return x0.flatten(start_dim=1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.0, **kwargs):
        super().__init__()
        attention_dim = attention_dim or input_dim
        self.proj = nn.Linear(input_dim, attention_dim)
        self.attn = nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x):
        x = self.proj(x)
        out, _ = self.attn(x, x, x)
        return out


class Phi_function(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq_state, final_state):
        final_state = final_state.unsqueeze(1).expand_as(seq_state)
        return self.net(torch.cat([seq_state, final_state], dim=-1)).squeeze(-1)


class HEA(nn.Module):
    def __init__(self, ple_arch, inp_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.experts = nn.ModuleList([MLP(ple_arch, inp_dim, dropout) for _ in range(2)])
        self.gate = nn.Linear(inp_dim, len(self.experts))

    def forward(self, x_list, ls_ratios=None):
        x = torch.cat(x_list, dim=-1) if isinstance(x_list, (list, tuple)) else x_list
        weights = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


class MultilevelMemoryHEA(nn.Module):
    def __init__(self, ple_arch, inp_dim, dropout=0.0, analysis_dim=None, **kwargs):
        super().__init__()
        self.inp_dim = inp_dim
        self.analysis_dim = analysis_dim or inp_dim
        self.base = HEA(ple_arch, inp_dim * 2, dropout)
        self.analysis_proj = nn.Linear(self.analysis_dim, inp_dim)

    def forward(self, x_list, multilevel_memory_data=None):
        if multilevel_memory_data is None or "analysis_vec" not in multilevel_memory_data:
            return self.base(x_list)

        # MTR conditions the semantic vectors before expert routing.
        analysis = self.analysis_proj(multilevel_memory_data["analysis_vec"])
        enhanced = []
        for x in x_list:
            enhanced.append(x + analysis)
        return self.base(enhanced)


class MultilevelMemoryMoE(nn.Module):
    def __init__(self, moe_arch, inp_dim, dropout=0.0, analysis_dim=None, **kwargs):
        super().__init__()
        self.expert = MLP(moe_arch, inp_dim, dropout)
        self.analysis_proj = nn.Linear(analysis_dim or inp_dim, inp_dim)

    def forward(self, x, multilevel_memory_data=None):
        if multilevel_memory_data is not None and "analysis_vec" in multilevel_memory_data:
            # MTR features condition the recommendation representation.
            x = x + self.analysis_proj(multilevel_memory_data["analysis_vec"])
        return self.expert(x)


class ConvertNet(nn.Module):
    def __init__(self, args, inp_dim, dropout=0.0, conv_type="HEA"):
        super().__init__()
        arch = getattr(args, "convert_arch", [128, 32])
        if conv_type == "MultilevelMemoryHEA":
            self.sub_module = MultilevelMemoryHEA(arch, inp_dim, dropout, analysis_dim=getattr(args, "analysis_vec_dim", inp_dim))
        else:
            self.sub_module = HEA(arch, inp_dim * 2, dropout)

    def forward(self, x_list, ls_ratios=None, multilevel_memory_data=None):
        if isinstance(self.sub_module, MultilevelMemoryHEA):
            return self.sub_module(x_list, multilevel_memory_data)
        return self.sub_module(x_list, ls_ratios)
