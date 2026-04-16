import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models_mamba import create_block, RMSNorm, rms_norm_fn, PACKET_NUM, StrideEmbed
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed
import math
from configuration_SPGE import SPGEConfig
from typing import Any, Dict, List, Optional, Tuple, Union



class CrossMamba(nn.Module):
    def __init__(self, embed_dim, crossmamba_depth):
        super(CrossMamba, self).__init__()
        self.crossmamba_depth = crossmamba_depth
        drop_path_rate = 0.1
        encoder_dpr = [x.item() for x in
                       torch.linspace(0, drop_path_rate, crossmamba_depth)]
        encoder_inter_dpr = [0.0] + encoder_dpr
        self.cross_mamba = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                if_bimamba=False,
                bimamba_type="v3",
                drop_path=encoder_inter_dpr[i],
                if_devide_out=True,
                init_layer_scale=None,
            )
            for i in range(crossmamba_depth)])
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3,
                                padding=1, groups=embed_dim)
        self.encoder_norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x_mamba, x_attn):
        residual = None
        hidden_states = x_mamba
        for blk in self.cross_mamba:
            hidden_states, residual = blk(hidden_states, residual=residual, extra_emb=x_attn)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.encoder_norm_f.weight,
            self.encoder_norm_f.bias,
            eps=self.encoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        return x


class SPGE_expert(nn.Module):
    def __init__(self, config: SPGEConfig, shared_gate_proj: nn.Linear, shared_up_proj: nn.Linear,
                 shared_down_proj: nn.Linear):
        super().__init__()
        self.private_expert_intermediate_dim = config.private_expert_intermediate_size
        self.hidden_dim = config.hidden_size


        self.shared_gate_proj = shared_gate_proj
        self.shared_up_proj = shared_up_proj
        self.shared_down_proj = shared_down_proj


        self.gate_proj = nn.Linear(
            self.hidden_dim,
            self.private_expert_intermediate_dim,
            bias=config.hidden_bias,
        )
        self.gate_act_fn = F.relu

        self.up_proj = nn.Linear(
            self.hidden_dim,
            self.private_expert_intermediate_dim,
            bias=config.hidden_bias,
        )
        self.up_act_fn = nn.Tanh()

        self.down_proj = nn.Linear(
            self.private_expert_intermediate_dim,
            self.hidden_dim,
            bias=config.hidden_bias,
        )
        self.private_expert_gate = nn.Linear(
            self.hidden_dim,
            1,
            bias=False,
        )
        # self.shared_expert_gate = nn.Linear(
        #     self.hidden_dim,
        #     1,
        #     bias=False,
        # )


    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:

        hidden_states = self.shared_down_proj(self.gate_act_fn(self.shared_gate_proj(hidden_states)))

        hidden_states = F.sigmoid(self.private_expert_gate(hidden_states)) * hidden_states

        return self.down_proj(self.gate_act_fn(self.gate_proj(hidden_states)))


class SPGEMoeLayer(nn.Module):


    def __init__(self, config: SPGEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.shared_expert_intermediate_dim = config.shared_expert_intermediate_size
        self.private_expert_intermediate_dim = config.private_expert_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_aux_loss = True


        self.router = nn.Linear(
            self.hidden_dim,
            self.num_experts,
            bias=False,
        )


        self.shared_gate_proj = nn.Linear(
            self.hidden_dim,
            self.shared_expert_intermediate_dim,
            bias=config.hidden_bias,
        )

        self.shared_up_proj = nn.Linear(
            self.hidden_dim,
            self.shared_expert_intermediate_dim,
            bias=config.hidden_bias,
        )

        self.shared_down_proj = nn.Linear(
            self.shared_expert_intermediate_dim,
            self.hidden_dim,
            bias=config.hidden_bias,
        )

        self.experts = nn.ModuleList(
            [SPGE_expert(config, self.shared_gate_proj, self.shared_up_proj, self.shared_down_proj) for
             _ in range(self.num_experts)])

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_states)
        routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue


            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        aux_loss = None
        if self.use_aux_loss:
            eps = 1e-9
            T, E = routing_probs.size(0), routing_probs.size(1)

            importance = routing_probs.sum(dim=0)

            load = expert_mask.sum(dim=(1, 2)).to(routing_probs.dtype)
            imp = importance / (importance.sum() + eps)
            ld = load / (load.sum() + eps)

            aux_loss = E * torch.sum(imp * ld)

        return final_hidden_states, router_logits, aux_loss

# todo: class MambaVisionMixer(nn.Module): -> class ETLDP_Mixer(nn.Module):
class ETLDP_Mixer(nn.Module):
    def __init__(
            self,
            config: SPGEConfig,
            embed_dim=256,
            encoder_depth=1,
            drop_path_rate=0.1,
            bimamba_type="none",
            depth=4,
            counter=1,
            num_experts=4,
    ):
        super().__init__()
        encoder_dpr = [x.item() for x in
                       torch.linspace(0, drop_path_rate, encoder_depth)]
        encoder_inter_dpr = [0.0] + encoder_dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.encoder_blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                if_bimamba=False,
                bimamba_type=bimamba_type,
                drop_path=encoder_inter_dpr[i],
                if_devide_out=True,
                init_layer_scale=None,
            )
            for i in range(encoder_depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        ffn_layer_class = SPGEMoeLayer
        self.feed_forward = ffn_layer_class(config)
        self.counter = counter
        self.depth = depth
        self.hidden_dropout = config.hidden_dropout
        self.CrossMamba = CrossMamba(embed_dim, crossmamba_depth=2)


        self.attn_norm = nn.LayerNorm(embed_dim)
        self.normy = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3,
                               padding=1, groups=embed_dim)

    def forward(self, x):
        residual = None
        hidden_states = x
        aux_loss = None

        for idx, blk in enumerate(self.encoder_blocks):
            hidden_states, residual = blk(hidden_states, residual)

            if self.counter == 0 and self.depth in [4, 8]:
                residual = self.CrossMamba(hidden_states, x)

        if self.counter == self.depth // 2 - 1:
            residual = hidden_states
            ff_outputs = self.feed_forward(hidden_states)
            if isinstance(ff_outputs, tuple):
                hidden_states, router_logits, aux_loss = ff_outputs
            else:
                hidden_states, router_logits, aux_loss = ff_outputs, None, None

            hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)


        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        if aux_loss:
            return x, aux_loss
        else:
            return x


class CA(nn.Module):
    def __init__(self, input_dim, num):
        super(CA, self).__init__()
        self.num = num
        self.multiattn = nn.ModuleList()
        self.ln = nn.ModuleList()
        for i in range(num):
            self.multiattn.append(nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True))
            if i != num - 1:
                self.ln.append(nn.LayerNorm(input_dim))

    def forward(self, tgt, memory):
        for i in range(self.num):
            tgt = tgt + self.multiattn[i](tgt, memory, memory)[0]
            if i != self.num - 1:
                tgt = self.ln[i](tgt)
        return tgt


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, counter=0, depth=0, first_half=None, **kwargs):
        super().__init__()
        self.counter = counter
        self.depth = depth
        self.first_half = first_half
        self.attn_norm = nn.LayerNorm(dim)
        self.normy = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = int(window / 10) - 1
        self.window = int(window)
        self.dwc_1D = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3,
                                padding=1, groups=dim)
        # self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, self.window))  # 从 (7, 7) 调整为
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, self.window))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, self.window))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, self.window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, self.window, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, self.window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        # pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool1d(output_size=agent_num)


        self.fusion_encoder = CA(input_dim=dim, num=2)

        drop_path_rate = 0.1
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        b, n, c = x.shape
        hidden_states = x
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        agent_tokens = self.pool(q[:, :-1, :].permute(0, 2, 1)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = self.an_bias
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = self.na_bias
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, :-1, :].transpose(1, 2).reshape(b, n - 1, c).permute(0, 2, 1)  # (B, C, L-1)
        x[:, :-1, :] = x[:, :-1, :] + self.dwc_1D(v_).permute(0, 2, 1)  # (B, L-1, C)

        if self.counter == sorted(self.first_half)[0]:
            x = hidden_states + self.drop_path(
                self.fusion_encoder(self.attn_norm(x), self.normy(hidden_states.transpose(1, 2)).transpose(1, 2)))


        x = self.proj(x)
        x = self.proj_drop(x)

        return x




class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            counter=0,
            depth=0,
            first_half={1},
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        self.counter = counter
        self.depth = depth
        self.first_half = first_half

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 depth,
                 transformer_blocks,
                 num_patches,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if depth == 1:
            # todo: self.mixer = MambaVisionMixer -> self.mixer = ETLDP_Mixer
            self.mixer = ETLDP_Mixer(config = SPGEConfig(), embed_dim=256, encoder_depth=1, counter=counter, depth=depth, num_experts=4)
        elif counter in transformer_blocks:
            half_len = len(transformer_blocks) // 2
            if half_len == 0:
                self.mixer = AgentAttention(
                    dim,
                    window=num_patches,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    counter=counter,
                    first_half={1},
                    depth=depth,
                    norm_layer=norm_layer,
                )
            else:
                first_half = set(transformer_blocks[:half_len])
                second_half = set(transformer_blocks[half_len:])
                if counter in first_half:
                    self.mixer = AgentAttention(
                        dim,
                        window=num_patches,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        counter=counter,
                        first_half=first_half,
                        depth=depth,
                        norm_layer=norm_layer,
                    )
                elif counter in second_half:
                    self.mixer = Attention(
                        dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        counter=counter,
                        depth=depth,
                        norm_layer=norm_layer,
                    )
        else:
            # todo: # todo: self.mixer = MambaVisionMixer -> self.mixer = ETLDP_Mixer
            self.mixer = ETLDP_Mixer(config=SPGEConfig(), embed_dim=256, encoder_depth=1, counter=counter, depth=depth, num_experts=4)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        out = self.mixer(self.norm1(x))
        if isinstance(out, tuple):
            hidden_states, aux_loss = out
        else:
            hidden_states, aux_loss = out, None
        x = x + self.drop_path(self.gamma_1 * hidden_states)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x, aux_loss


# todo: class MambaVisionLayer(nn.Module): - > class ETLDP_Layer(nn.Module)
class ETLDP_Layer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 num_patches,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):

        super().__init__()

        self.blocks = nn.ModuleList([Block(dim=dim,
                                           counter=i,
                                           depth=depth,
                                           transformer_blocks=transformer_blocks,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           num_patches=num_patches,
                                           qk_scale=qk_scale,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           layer_scale=layer_scale)
                                     for i in range(depth)])
        print(transformer_blocks)

    def forward(self, x):
        total_aux_loss = 0.0
        for _, blk in enumerate(self.blocks):
            x, aux_loss = blk(x)
            if aux_loss is not None:
                total_aux_loss += aux_loss

        return x, total_aux_loss

# todo: class MambaVision(nn.Module): -> class ETLDP_Core(nn.Module):
class ETLDP_Core(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 num_patches,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):

        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            # todo: self.levels.append(level) 中的level = MambaVisionLayer -> level = ETLDP_Layer
            level = ETLDP_Layer(dim=dim,
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     num_patches=num_patches,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(
                                         range(depths[i] // 2, depths[i])),
                                     )
            self.levels.append(level)

        self.norm = nn.BatchNorm2d(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward(self, x):
        total_aux_loss = 0.0
        for level in self.levels:
            x, aux_loss = level(x)
            if aux_loss is not None:
                total_aux_loss += aux_loss
        return x, total_aux_loss


# todo: class NetMamba(nn.Module) -> class ETLDP(nn.Module):
class ETLDP(nn.Module):
    def __init__(self, img_size=40, stride_size=4, in_chans=1,
                 dim=256,
                 embed_dim=256,
                 decoder_embed_dim=128, decoder_depth=2,
                 num_classes=1000,
                 norm_pix_loss=False,
                 drop_path_rate=0.1,
                 bimamba_type="none",
                 is_pretrain=False,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.is_pretrain = is_pretrain

        self.patch_embed = StrideEmbed(img_size, img_size, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # todo: class NetMamba(nn.Module): 中的 self.mambavision -> self.etldp_core_block
        self.etldp_core_block = ETLDP_Core(depths=[1, 2, 4, 8],
                                       num_heads=[8, 8, 8, 8],
                                       window_size=[8, 8, 14, 7],
                                       dim=256,
                                       num_patches=self.num_patches,
                                       in_dim=32,
                                       mlp_ratio=4,
                                       drop_path_rate=0.2,
                                       **kwargs)

        if is_pretrain:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, decoder_embed_dim))
            decoder_dpr = [x.item() for x in
                           torch.linspace(0, drop_path_rate, decoder_depth)]
            decoder_inter_dpr = [0.0] + decoder_dpr
            self.decoder_blocks = nn.ModuleList([
                create_block(
                    decoder_embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    if_bimamba=True,
                    bimamba_type=bimamba_type,
                    drop_path=decoder_inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                )
                for i in range(decoder_depth)])
            self.decoder_norm_f = RMSNorm(decoder_embed_dim, eps=1e-5)
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        trunc_normal_(self.pos_embed, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def stride_patchify(self, imgs, stride_size=4):
        B, C, L = imgs.shape
        assert C == 1, "Input should be 1 dim"
        x = imgs.reshape(B, L // stride_size, stride_size)
        return x

    def random_masking(self, x, mask_ratio):

        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, if_mask=True, ):
        B, C, L = x.shape
        x = self.patch_embed(x.reshape(B, C, L))
        x = x + self.pos_embed[:, :-1, :]
        if if_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)

        hidden_states = x
        x, total_aux_loss = self.etldp_core_block(hidden_states)

        if if_mask:
            return x, mask, ids_restore, total_aux_loss
        else:
            return x, total_aux_loss


    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        visible_tokens = x[:, :-1, :]
        x_ = torch.cat([visible_tokens, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x_, x[:, -1:, :]], dim=1)

        x = x + self.decoder_pos_embed

        residual = None
        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states, residual = blk(hidden_states, residual)
        fused_add_norm_fn = rms_norm_fn
        x = fused_add_norm_fn(
            self.drop_path(hidden_states),
            self.decoder_norm_f.weight,
            self.decoder_norm_f.bias,
            eps=self.decoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        x = self.decoder_pred(x)
        x = x[:, :-1, :]
        return x



    def forward_rec_loss(self, imgs, pred, mask):
        target = self.stride_patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss


    def forward(self, imgs, mask_ratio=0.9, ):
        B, C, L = imgs.shape
        assert C == 1, "Input should be 1 dim"
        if self.is_pretrain:
            latent, mask, ids_restore, total_aux_loss = self.forward_encoder(imgs, mask_ratio=mask_ratio, )
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_rec_loss(imgs, pred, mask)
            return loss, pred, mask, total_aux_loss
        else:
            x, total_aux_loss = self.forward_encoder(imgs, mask_ratio=mask_ratio, if_mask=False)
            return self.head(x[:, -1, :]), total_aux_loss


# todo: def net_mamba_pretrain(**kwargs): -> def etldp_pretrain(**kwargs):
def etldp_pretrain(**kwargs):
    model = ETLDP(
        img_size=40, stride_size=4, in_chans=1,
        dim=256, depth=6,
        decoder_embed_dim=128, decoder_depth=2,
        drop_path_rate=0.1,
        is_pretrain=True,
        device=None, dtype=None,
        **kwargs)
    return model

# todo: def net_mamba_classifier(**kwargs): -> def etldp_classifier(**kwargs):
def etldp_classifier(**kwargs):
    model = ETLDP(
        img_size=40, stride_size=4, in_chans=1,
        dim=256, depth=6,
        decoder_embed_dim=128, decoder_depth=2,
        is_pretrain=False,
        device=None, dtype=None,
        **kwargs)
    return model