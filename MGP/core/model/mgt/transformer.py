import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from torch.distributions import Categorical
from .pose_encoding import PositionEmbedding
from diffusion_policy_3d.model.mgt_utils.utils_model import generate_src_mask, gumbel_sample, cosine_schedule, linear_schedule

class SelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_mask is not None:
            att[~src_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class SelfAttn_Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), src_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, 77)).view(1, 1, block_size, 77))
        self.n_head = n_head

    def forward(self, x, word_emb):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CrossAttn_Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, word_emb):
        x = x + self.attn(self.ln1(x), self.ln3(word_emb))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):

    def __init__(self,
                 vqvae,
                 num_vq=1024,
                 embed_dim=512,
                 comb_state_dim=4,
                 cond_length=4,
                 block_size=16,
                 num_layers=2,
                 num_local_layer=1,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        # self.tok_emb = nn.Embedding(num_vq + 3, embed_dim).requires_grad_(False)
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.code_dim)  # [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb = nn.Linear(self.vqvae.code_dim, embed_dim)
        # print('state_dim:', state_dim)
        self.cond_emb_origin = nn.Linear(comb_state_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[SelfAttn_Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in
                                      range(num_layers - num_local_layer)])
        self.pos_embed = PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.cond_pos_embed = PositionEmbedding(cond_length, embed_dim, 0.0, False)

        self.num_local_layer = num_local_layer
        if num_local_layer > 0:
            # self.word_emb = nn.Linear(clip_dim, embed_dim)
            self.comb_state_emb = nn.Linear(comb_state_dim, embed_dim)
            # self.pc_emb = nn.Linear(pc_dim, embed_dim)
            self.cross_att = nn.Sequential(
                *[CrossAttn_Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in
                  range(num_local_layer)])
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, src_mask, comb_state):
        # if len(idx) == 0:
        #     token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        # else:
        b, t = idx.size()
        # print('idx:', idx.shape)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the Trans model
        not_learn_idx = idx < self.vqvae.nb_code
        learn_idx = ~not_learn_idx
        # print('not_learn_idx:', not_learn_idx[0])
        # print('not_learn_idx:', idx[not_learn_idx].shape)
        token_embeddings = torch.empty((*idx.shape, self.vqvae.code_dim), device=idx.device)
        token_embeddings[not_learn_idx] = self.vqvae.quantizer.dequantize(idx[not_learn_idx]).requires_grad_(
            False)
        computed_index = idx[learn_idx] - self.vqvae.nb_code
        # print("Debug: idx[learn_idx] =", idx[learn_idx].cpu().numpy())
        # print("Debug: self.vqvae.nb_code =", self.vqvae.nb_code)
        # print("Debug: computed index =", computed_index.cpu().numpy())

        # Then do the lookup:
        token_embeddings[learn_idx] = self.learn_tok_emb(computed_index)

        # token_embeddings[learn_idx] = self.learn_tok_emb(idx[learn_idx] - self.vqvae.nb_code)
        token_embeddings = self.to_emb(token_embeddings)

        if self.num_local_layer > 0:
            comb_state_emb = self.comb_state_emb(comb_state)
            # pc_emb = self.pc_emb(pc)
            # cond_emb = self.cond_pos_embed(torch.cat([state_emb, pc_emb], dim=1))
            token_embeddings = self.pos_embed(token_embeddings)
            for module in self.cross_att:
                token_embeddings = module(token_embeddings, comb_state_emb)
        # token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
        # print('token_embeddings:', token_embeddings.shape)
        x = self.pos_embed(token_embeddings)
        for block in self.blocks:
            x = block(x, src_mask)

        return x
'''
Base: first looks at each token in idx to see whether from vqvar or of special tokens;
for tokens from vqvae, 'dequantize' them; for special tokens, use separate learned emb;
then passed through linear layer and then processed with positional emb;
fuse state through cross-attention
a stack of self-attetnion block is applied
'''


class CrossCondTransHead(nn.Module):

    def __init__(self,
                num_vq=1024,
                embed_dim=512,
                block_size=16,
                num_layers=2,
                n_head=8,
                drop_out_rate=0.1,
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[SelfAttn_Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

'''
Head process: 
after a final layer norm, it prohects the representation with a linear layer into logits for each position
[bs,l,num_vq]
'''

class ActTransformer(nn.Module):
    def __init__(self,
                 vqvae,
                 num_vq=1024,
                 embed_dim=512,
                 comb_state_dim=64,
                 cond_length=4,
                 block_size=16,
                 num_layers=2,
                 num_local_layer=0,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, comb_state_dim, cond_length, block_size, num_layers, num_local_layer,
                                             n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        # print('parameter',sum(p.numel() for p in self.parameters()))
    def get_attn_mask(self, src_mask, att_txt=None):
        # if att_txt is None:
        #     att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        #     print('att_txt:', att_txt.shape)
        # src_mask = torch.cat([att_txt, src_mask],  dim=1)
        # print('src_mask:', src_mask.shape)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward(self, idx, src_mask, comb_state):
        # idx [bs,L]; src_mask [bs,L]; state[bs,state_dim]
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask)
        # print('src_mask in act:', src_mask.shape)
        feat = self.trans_base(idx, src_mask, comb_state)
        # trans_base: converts a sequence of discrete codes into a seq of continuous vectors that carry both token information
        # and conditioning information
        logits = self.trans_head(feat, src_mask)
        # trans_head: The result is a tensor of logits (unnormalized probabilities) over the vocabulary for every position in the sequence.
        return logits

    def fast_sample_firsttoken(self, first_tokens, src_mask, comb_state, m_length=12, step=1, gt=None):
        '''
        :param first_tokens: token to start with (batch_size, 1)
        :param src_mask: token mask (batch_size, block_size)
        :param state: condition state (batch_size, state_t, state_dim)
        :param pc: condition point cloud (batch_size, pc_t, pc_dim)
        :param m_length: max length of action sequence (batch_size)
        :param step: max step to sample (int)
        :return: ids (batch_size, block_size - 1)
        '''
        assert len(first_tokens.shape) == 1
        batch_size = comb_state.shape[0]
        rand_pos = True
        pad_id = self.num_vq + 1
        mask_id = self.num_vq + 2
        shape = (batch_size, self.block_size - 1)
        if m_length is None:
            m_length = torch.full((batch_size,), 12, dtype=torch.long, device=comb_state.device)
        m_tokens_len = torch.ceil((m_length) / 4).long()
        scores = torch.ones(shape, dtype=torch.float32, device=comb_state.device)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)

        ids = torch.full(shape, mask_id, dtype=torch.long, device=comb_state.device)

        # ids[:, 0] = first_tokens
        sample_max_steps = torch.round(step / m_length * m_tokens_len) + 1e-8
        '''
        need to check!!!
        '''
        # if src_mask is not None:
        #         src_mask = self.get_attn_mask(src_mask)
        for i in range(step):
            # if src_mask is not None:
            #     src_mask = self.get_attn_mask(src_mask)

            timestep = torch.clip(step / (sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            scores[~src_token_mask_noend] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token
            sorted, sorted_score_indices = scores.sort(descending=True)  # deterministic
            
            ids[~src_token_mask] = self.num_vq + 1  # pad_id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.num_vq)  # [INFO] replace with end id         
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
          
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)           
            ids.scatter_(-1, sorted_score_indices, mask_id)          
            ids[:, 0] = first_tokens
            logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]
            filtered_logits = logits  # top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
            # print('ids after:', ids[:,0])
            # print('gt:', gt[:,0])
            # ids after: tensor([3302, 1355, 6743, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
            # 8193, 8193, 8193], device='cuda:0')
            # gt: tensor([3302, 6085, 1868, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
            # 8193, 8193, 8193, 8193], device='cuda:0', dtype=torch.int32)

        return ids

    def fast_sample(self, src_mask, comb_state, m_length=8, step=1, gt=None):
        '''
        :param first_tokens: token to start with (batch_size, 1)
        :param src_mask: token mask (batch_size, block_size)
        :param state: condition state (batch_size, state_t, state_dim)
        :param pc: condition point cloud (batch_size, pc_t, pc_dim)
        :param m_length: max length of action sequence (batch_size)
        :param step: max step to sample (int)
        :return: ids (batch_size, block_size - 1)
        '''
        batch_size = comb_state.shape[0]
        rand_pos = True
        pad_id = self.num_vq + 1
        mask_id = self.num_vq + 2
        shape = (batch_size, self.block_size - 1)
        if m_length is None:
            m_length = torch.full((batch_size,), 8, dtype=torch.long, device=comb_state.device)
        else:
            m_length = torch.full((batch_size,), m_length, dtype=torch.long, device=comb_state.device)
        m_tokens_len = torch.ceil((m_length) / 4).long()
        scores = torch.ones(shape, dtype=torch.float32, device=comb_state.device)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)

        ids = torch.full(shape, mask_id, dtype=torch.long, device=comb_state.device)

        # ids[:, 0] = first_tokens
        sample_max_steps = torch.round(step / m_length * m_tokens_len) + 1e-8
        '''
        need to check!!!
        '''
        # if src_mask is not None:
        #         src_mask = self.get_attn_mask(src_mask)
        for i in range(step):
            # if src_mask is not None:
            #     src_mask = self.get_attn_mask(src_mask)

            timestep = torch.clip(step / (sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1) 
            scores[~src_token_mask_noend] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token
            sorted, sorted_score_indices = scores.sort(descending=True)  # deterministic
            ids[~src_token_mask] = self.num_vq + 1  # pad_id

            ids.scatter_(-1, m_tokens_len[..., None].long(), self.num_vq)  # [INFO] replace with end id      
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)           
            ids.scatter_(-1, sorted_score_indices, mask_id)          
            # ids[:, 0] = first_tokens
            logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]
            filtered_logits = logits  # top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)
            # print('ids after:', ids[:,0])
            # print('gt:', gt[:,0])
            # ids after: tensor([3302, 1355, 6743, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
            # 8193, 8193, 8193], device='cuda:0')
            # gt: tensor([3302, 6085, 1868, 8192, 8193, 8193, 8193, 8193, 8193, 8193, 8193, 8193,
            # 8193, 8193, 8193, 8193], device='cuda:0', dtype=torch.int32)

        return ids


    def regressive_sample(self, tokens, src_mask, act_t, comb_state, m_length=12, step=1,  gt=None):
        '''
        :param first_tokens: token to start with (batch_size, 1)
        :param src_mask: token mask (batch_size, block_size)
        :param state: condition state (batch_size, state_dim)
        :param m_length: max length of action sequence (batch_size)
        :param step: max step to sample (int)
        :return: ids (batch_size, block_size - 1)
        '''
        # print('t:', act_t)
        # assert len(tokens.shape) == 1
        assert act_t % 4 == 0
        start_t = int(act_t/4)
        mask_id = self.num_vq + 2
        rand_pos = True
        pad_id = self.num_vq + 1

        batch_size = comb_state.shape[0]
        shape = (batch_size, self.block_size - 1)
        # print('block_size', self.block_size)
        # print('m_length', m_length)
        ids = torch.full(shape, mask_id, dtype=torch.long, device=tokens.device)
        ids[:, :start_t] = tokens[:, :start_t]

        if m_length is None:
            m_length = torch.full((batch_size,), 12, dtype=torch.long, device=comb_state.device)
        else:
            m_length = torch.full((batch_size,), m_length, dtype=torch.long, device=comb_state.device)

        m_tokens_len = torch.ceil((m_length) / 4).long()
        scores = torch.ones(shape, dtype=torch.float32, device=comb_state.device)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)
        sample_max_steps = torch.round(step / m_length * m_tokens_len) + 1e-8

        second_partition_pos_idx = m_tokens_len - start_t

        for i in range(step):
            # if src_mask is not None:
            #     print('src_mask:', src_mask.shape)
            #     src_mask = self.get_attn_mask(src_mask)

            timestep = torch.clip(step / (sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            scores[~src_token_mask_noend] = 0
            scores[:, :start_t] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token
            sorted, sorted_score_indices = scores.sort(descending=True)  # deterministic

            ids[:, :start_t] = tokens[:, :start_t]
            ids[~src_token_mask] = self.num_vq + 1  # pad_id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.num_vq)  # [INFO] replace with end id
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)

            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            # print('ids:', ids)
            # print('mask:', src_token_mask)
            # print('comb_state:', comb_state.shape)
            # print('ids shape', ids.shape)
            # print('ids', ids)
            logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]

            filtered_logits = logits  # top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            ########
            ids[:, :start_t] = tokens[:, :start_t]
            ########

            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, 0)

        return ids


    def regressive_sample_new(self, tokens, src_mask, act_t, comb_state, scores, m_length=12, step=1):
        '''
        :param first_tokens: token to start with (batch_size, 1)
        :param src_mask: token mask (batch_size, block_size)
        :param state: condition state (batch_size, state_dim)
        :param m_length: max length of action sequence (batch_size)
        :param step: max step to sample (int)
        :return: ids (batch_size, block_size - 1)
        '''
        # assert len(tokens.shape) == 1
        assert act_t % 4 == 0
        start_t = int(act_t/4)
        mask_id = self.num_vq + 2
        rand_pos = True
        pad_id = self.num_vq + 1
        batch_size = comb_state.shape[0]
        shape = (batch_size, self.block_size - 1)
        # ids = torch.full(shape, mask_id, dtype=torch.long, device=tokens.device)
        step = 2
        ids = tokens
        ids[:, :start_t] = tokens[:, :start_t]

        if m_length is None:
            m_length = torch.full((batch_size,), 12, dtype=torch.long, device=comb_state.device)
        else:
            m_length = torch.full((batch_size,), m_length, dtype=torch.long, device=comb_state.device)

        m_tokens_len = torch.ceil((m_length) / 4).long()
        scores = scores.to(comb_state.device)
        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)
        sample_max_steps = torch.round(8 * step / m_length * m_tokens_len) + 1e-8
        second_partition_pos_idx = m_tokens_len - start_t
        for i in range(step):
            rand_mask_prob = linear_schedule(i, step, start_prob=0.7, end_prob=0.0)
            num_token_masked = (rand_mask_prob * second_partition_pos_idx).long().clip(min=1)
            scores[~src_token_mask_noend] = 0

            scores[:, :start_t] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token
            sorted, sorted_score_indices = scores.sort(descending=True)  # deterministic
            ids[:, :start_t] = tokens[:, :start_t]
            ids[~src_token_mask] = self.num_vq + 1  # pad_id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.num_vq)  # [INFO] replace with end id
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]

            filtered_logits = logits  # top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            ########
            ids[:, :start_t] = tokens[:, :start_t]
            ########

            probs_without_temperature = logits.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            if i < step-1:
                scores = scores.masked_fill(~is_mask, 0)

        return ids, scores


    def regressive_sample_ph(self, tokens, src_mask, act_t, comb_state, scores, m_length=12, step=1):
        '''
        :param first_tokens: token to start with (batch_size, 1)
        :param src_mask: token mask (batch_size, block_size)
        :param state: condition state (batch_size, state_dim)
        :param m_length: max length of action sequence (batch_size)
        :param step: max step to sample (int)
        :return: ids (batch_size, block_size - 1)
        '''
        # assert len(tokens.shape) == 1
        assert act_t % 4 == 0
        start_t = int(act_t/4)
        mask_id = self.num_vq + 2
        rand_pos = True
        pad_id = self.num_vq + 1
        batch_size = comb_state.shape[0]
        shape = (batch_size, self.block_size - 1)
        scores = torch.ones(shape, dtype=torch.float32, device=comb_state.device)
        # ids = torch.full(shape, mask_id, dtype=torch.long, device=tokens.device)
        step = 2
        ids = tokens
        ids[:, :start_t] = tokens[:, :start_t]

        if m_length is None:
            m_length = torch.full((batch_size,), 12, dtype=torch.long, device=comb_state.device)
        else:
            m_length = torch.full((batch_size,), m_length, dtype=torch.long, device=comb_state.device)

        m_tokens_len = torch.ceil((m_length) / 4).long()

        src_token_mask = generate_src_mask(self.block_size - 1, m_tokens_len + 1)
        src_token_mask_noend = generate_src_mask(self.block_size - 1, m_tokens_len)
        sample_max_steps = torch.round(8 * step / m_length * m_tokens_len) + 1e-8
        second_partition_pos_idx = m_tokens_len - start_t

        init_logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]
        init_ids = gumbel_sample(init_logits, temperature=0, dim=-1)
        probs_without_temperature = init_logits.softmax(dim=-1)
        scores = 1 - probs_without_temperature.gather(-1, init_ids[..., None])
        scores = rearrange(scores, '... 1 -> ...')

        for i in range(step):
            rand_mask_prob = linear_schedule(i, step, start_prob=0.7, end_prob=0.0)
            num_token_masked = (rand_mask_prob * second_partition_pos_idx).long().clip(min=1)
            scores[~src_token_mask_noend] = 0

            scores[:, :start_t] = 0
            scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token
            sorted, sorted_score_indices = scores.sort(descending=True)  # deterministic
            ids[:, :start_t] = tokens[:, :start_t]
            ids[~src_token_mask] = self.num_vq + 1  # pad_id
            ids.scatter_(-1, m_tokens_len[..., None].long(), self.num_vq)  # [INFO] replace with end id
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            logits = self.forward(idx=ids, src_mask=src_token_mask, comb_state=comb_state)[:, 0:]

            filtered_logits = logits  # top_p(logits, .5) # #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0  # starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            is_mask = ids == mask_id
            ids = torch.where(is_mask, pred_ids, ids)

            ########
            ids[:, :start_t] = tokens[:, :start_t]
            ########

            probs_without_temperature = logits.softmax(dim=-1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            if i < step-1:
                scores = scores.masked_fill(~is_mask, 0)

        return ids, scores

'''
first token provided and all others mark
ed as masked
loop: decide how many tokens are uncertain (using cosine schedule);
    identify the most uncertain positions (using scores);
    force those positions to remain masked;
    pass the current sequence(with some masked tokens) through the transformer;
    sample new tokens for the masked positions;
    replace the masked positions with new tokens;
    update 'incertainty' score based on the model's confidence
final output [bs, block_size-1]
'''