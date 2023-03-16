# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------


import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from .builder import MODELS
from .misc import Result
from .utils import ResidualAttentionBlock
from ipdb import set_trace
import clip
from transformers import AutoModel
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        proj_std = (self.width**-0.5) * ((2 * self.layers)**-0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width)**-0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        for i, resblock in enumerate(self.resblocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(resblock, x)
            else:
                x = resblock(x)
        return x

@MODELS.register_module()
class DistilBert(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        pretrained=True,
        fixed=True,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
        self.transformer.train()
        self.width = width
    
        if fixed is True:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if pretrained is False:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, as_dict=True):
        outs = Result(as_dict=as_dict)
        out_x = self.transformer(**x)
        out_hidden = out_x.last_hidden_state[:, 0, :]
        last_hidden = out_x.hidden_states[-1]

        outs.append(out_hidden, name='x')
        outs.append(last_hidden, name='all_tokens')
        return outs.as_return()

@MODELS.register_module()
class Bert(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        pretrained=True,
        fixed=True,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.transformer.train()
        self.width = width
    
        if fixed is True:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if pretrained is False:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, as_dict=True):
        outs = Result(as_dict=as_dict)
        out_x = self.transformer(**x)
        out_hidden = out_x.last_hidden_state[:, 0, :]
        last_hidden = out_x.hidden_states[-1]

        outs.append(out_hidden, name='x')
        outs.append(last_hidden, name='all_tokens')
        return outs.as_return()
    
@MODELS.register_module()
class Roberta(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        pretrained=True,
        fixed=True,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('roberta-base', output_hidden_states=True, cache_dir='/mnt/petrelfs/xujilan/checkpoints/')
        self.transformer.train()
        self.width = width
    
        if fixed is True:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if pretrained is False:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, question=None, as_dict=True):
        outs = Result(as_dict=as_dict)
        out_x = self.transformer(**x)
        out_hidden = out_x.last_hidden_state[:, 0, :]
        last_hidden = out_x.hidden_states[-1]

        outs.append(out_hidden, name='x')
        outs.append(last_hidden, name='all_tokens')
        return outs.as_return()

@MODELS.register_module()
class BertMedium(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        pretrained=True,
        fixed=True,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('prajjwal1/bert-medium', output_hidden_states=True)
        self.transformer.train()
        self.width = width
    
        if fixed is True:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if pretrained is False:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, as_dict=True):
        outs = Result(as_dict=as_dict)
        out_x = self.transformer(**x)
        out_hidden = out_x.last_hidden_state[:, 0, :]
        last_hidden = out_x.hidden_states[-1]

        outs.append(out_hidden, name='x')
        outs.append(last_hidden, name='all_tokens')
        return outs.as_return()

@MODELS.register_module()
class TextTransformer(nn.Module):

    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        pretrained=True,
        fixed=True,
    ):

        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),
            use_checkpoint=use_checkpoint)

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        clip_model, _ = clip.load('ViT-B/16', device='cuda', jit=False)
        self.text_projection = nn.Parameter(torch.empty(clip_model.text_projection.shape))
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        # initialization
        nn.init.normal_(self.positional_embedding, std=0.01)

        if pretrained:
            print('loading clip weights for text encoder')
            self.reload_clip_weights(clip_model)
        if fixed:
            print('freezing text encoder')
            self.freeze_text_encoder()

    def freeze_text_encoder(self):
        for p in self.parameters():
            p.requires_grad=False

    def reload_clip_weights(self, clip_model):
        text_dict = clip_model.state_dict()
        msg = self.load_state_dict(text_dict, strict=False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, *, as_dict=True):
        x = self.token_embedding(text)
        outs = Result(as_dict=as_dict)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        ### w/o text projection ###
        # all_tokens = x.clone()
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        ### w/ text projection ###
        all_tokens = x.clone() @ self.text_projection
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        outs.append(x, name='x')
        outs.append(all_tokens, name='all_tokens')

        return outs.as_return()
