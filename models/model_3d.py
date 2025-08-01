import timm
import torch
from torch import nn


class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim // 2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        # aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)  # (batch_size, input_dim)
        return weighted_instances


class ConvNextLSTM(nn.Module):
    def __init__(self, in_chans=3, class_num=1):
        super(ConvNextLSTM, self).__init__()

        self.backbone = 'convnext_small.fb_in22k_ft_in1k_384'
        backbone = timm.create_model(self.backbone, pretrained=False, in_chans=in_chans, global_pool='', num_classes=0)

        self.encoder = backbone

        num_features = self.encoder.num_features
        # self.bn2 = nn.BatchNorm2d(num_features)

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))

        self.lstm = LSTMMIL(
            num_features)  # nn.LSTM(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        # self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)

        self.head = nn.Sequential(
            # nn.Linear(lstm_embed, lstm_embed//2),
            # nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            # nn.LeakyReLU(0.1),
            nn.Linear(num_features, class_num),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs, in_chans, n_slice_per_c, image_size, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, in_chans, image_size, image_size)
        x = self.encoder.forward_features(x)
        x = self.flatten(x)
        x = x.contiguous().reshape(bs, n_slice_per_c, -1)
        x = self.lstm(x)
        # feat = feat.contiguous().view(bs, n_slice_per_c, -1)
        # feat = torch.mean(feat, dim=1)
        x = self.head(x)
        # feat = feat.view(bs, -1).contiguous()
        # print(feat.shape)
        return x


if __name__ == "__main__":

    model = ConvNextLSTM(in_chans=1)
    x = torch.rand(1, 1, 64, 64, 64)
    out = model(x)
    print(out.shape)
