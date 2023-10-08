import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = 0.5*temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.avgpool = nn.AvgPool2d((1,2),stride=(1,2))

    def forward(self, q, k, v, mask=None):
        k = self.avgpool(k)
        v = self.avgpool(v)

        attn = torch.matmul(q.transpose(2,3) / self.temperature, k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v.transpose(2,3))

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 3).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual
        return q, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(self.Mish(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.weight = nn.Parameter(torch.ones(128),requires_grad=True)
        self.avg_pool = nn.AvgPool2d((1,2),stride=(1,2))
        self.Linear1 = nn.Linear(384, 128)
        self.Linear2 = nn.Linear(256, 128)
        self.softmax = nn.Softmax(dim=- 1)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        enc_output = src_seq
        m = 1

        ################################
        m1 = enc_output[:, :, 0:128]
        m2 = enc_output[:, :, 128:]
        for enc_layer in self.layer_stack:
            m1, enc_slf_attn = enc_layer(m1, m1, m1, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        for enc_layer in self.layer_stack:
            m2, enc_slf_attn = enc_layer(m2, m2, m2, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []


        t=128
        l1 = 0
        # EEG→PPS      # N层
        enc_output1 = enc_output[:, :, t:]
        k1 = enc_output[:, :, 0:t]
        v1 = enc_output[:, :, 0:t]
        for enc_layer in self.layer_stack:
            if l1 < m:
                enc_output1, enc_slf_attn = enc_layer(enc_output1, k1, v1, slf_attn_mask=src_mask)
            else:
                enc_output1, enc_slf_attn = enc_layer(enc_output1, enc_output1, enc_output1, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            l1 += 1

        l2 = 0
        # PPS→EEG      # N层
        enc_output2 = enc_output[:, :, 0:t]
        k2 = enc_output[:, :, t:]
        v2 = enc_output[:, :, t:]
        for enc_layer in self.layer_stack:
            if l2 < m:
                enc_output2, enc_slf_attn = enc_layer(enc_output2, k2, v2, slf_attn_mask=src_mask)
            else:
                enc_output2, enc_slf_attn = enc_layer(enc_output2, enc_output2, enc_output2, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            l2 += 1

        enc_output = torch.cat((enc_output1, enc_output2), dim=2)
        enc_output = self.Linear2(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, enc_output, enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output, m1, m2


class CorruptionLayer(nn.Module):
    def __init__(self, device, corrupt_probability=0.1):
        super(CorruptionLayer, self).__init__()
        self.corrupt_p = corrupt_probability
        self.device = device

    def forward(self, feature):
        bitmask = torch.cuda.FloatTensor(feature.shape).uniform_() > self.corrupt_p
        return torch.mul(feature, bitmask)

class TransformerEncoder(nn.Module):

    def __init__(self,sentence_len,  d_feature, n_layers=6, n_heads=8, p_drop=0.1, d_ff=2048):
        super(TransformerEncoder, self).__init__()
        d_k = d_v = d_feature // n_heads
        self.encoder = Encoder( d_model=d_feature, d_inner=d_ff,
                               n_layers=n_layers, n_head=n_heads, d_k=d_k, d_v=d_v,
                               dropout=p_drop)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=- 1)

        # 通道注意力 channel-wise attention
        self.avgPool = nn.AvgPool2d((1, 128), stride=(1, 128))

        self.L1_eeg = nn.Linear(32, 64)
        self.L2_eeg = nn.Linear(64, 32)

        self.L1_pps = nn.Linear(8, 12)
        self.L2_pps = nn.Linear(12, 8)

        self.Mish = nn.Mish()

        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.ones((32,2)), requires_grad=True)

        self.softmax1 = nn.Softmax(dim=-1)

        self.EEG_temp = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 16), stride=(2, 8), padding=0, bias=False))
        self.EEG_DWS_Conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(6, 6), stride=(3, 3), padding=0, bias=False, groups=16),
            nn.AvgPool2d((2,2),stride=(2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False))

        self.PPS_temp = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 16), stride=(1, 8), padding=0, bias=False))
        self.PPS_DWS_Conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 6), stride=(2, 3), padding=0, bias=False, groups=16),
            nn.AvgPool2d((2, 2),stride=(2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False))

    def forward(self, src_seq):
        src_mask = None
        SRC_seq = None

        for batch in range(src_seq.shape[0]):
            batch_EEG = src_seq[batch, :, 0:32, :]
            batch_EEG_temp = self.EEG_temp(batch_EEG)
            batch_EEG_DWS = self.EEG_DWS_Conv(batch_EEG_temp)
            batch_EEG_flatten = torch.flatten(batch_EEG_DWS, start_dim=0, end_dim=-1)

            batch_PPS = src_seq[batch, :, 32:, :]
            batch_PPS_temp = self.PPS_temp(batch_PPS)
            batch_PPS_DWS = self.PPS_DWS_Conv(batch_PPS_temp)
            batch_PPS_flatten = torch.flatten(batch_PPS_DWS, start_dim=0, end_dim=-1)

            batch_concat = torch.cat((batch_EEG_flatten, batch_PPS_flatten))

            if SRC_seq is None:
                SRC_seq = batch_concat
            else:
                SRC_seq = torch.vstack((SRC_seq, batch_concat))

        SRC_seq = torch.unsqueeze(SRC_seq, 1)

        outputs_feature, outputs_m1, outputs_m2 = self.encoder(SRC_seq, src_mask)

        outputs, _ = torch.max(outputs_feature, dim=1)
        outputs_classification = self.softmax(self.linear(outputs))

        outputs_m1, _ = torch.max(outputs_m1, dim=1)
        outputs_classification_m1 = self.softmax(self.linear(outputs_m1))

        outputs_m2, _ = torch.max(outputs_m2, dim=1)
        outputs_classification_m2 = self.softmax(self.linear(outputs_m2))

        return outputs_feature, outputs_classification,  outputs_classification_m1, outputs_classification_m2
