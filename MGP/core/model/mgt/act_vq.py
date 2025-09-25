import torch.nn as nn
from .resnet import Resnet1D
from .quantization import QuantizeEMAReset


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=16,
                 down_t=3,
                 stride_t=2,
                 width=16,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=16,
                 down_t=3,
                 stride_t=2,
                 width=16,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)



class ActVQ(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=16,
                 output_emb_width=16,
                 down_t=3,
                 stride_t=2,
                 width=16,
                 depth=3,
                 dilation_growth_rate=2,
                 activation='relu',
                 norm=None):
        super().__init__()
        self.code_dim = 16
        self.nb_code = nb_code
        output_dim = 4
        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width,
                               depth, dilation_growth_rate, activation=activation,
                               norm=norm)
        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t, width,
                               depth, dilation_growth_rate, activation=activation,
                               norm=norm)

        self.quantizer = QuantizeEMAReset(nb_code, code_dim, args) # discretize the latent representations.


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        # print('emb shape:', x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def decode(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

    def preprocess(self, x):
        #  (bs,T,4) - (bs,4,T)
        x = x.permute(0, 2, 1).float()
        return x


    def postprocess(self, x):
        # (batch, 4, T) → Output: (batch, T, 4).
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        x_in = self.preprocess(x)     # (b,64,4) - (batch, 4, 64). 
        x_encoder = self.encoder(x_in)   # (b,c,T)
        # x_encoder = self.postprocess(x_encoder) # (b,T,c)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized) # (b,4,64)
        x_out = self.postprocess(x_decoder) # (b,64,4)
        return x_out, loss, perplexity
