import torch
import torch.nn as nn
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(self.pool1(x))
        return x

class EncoderBlock_Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock_Out, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class EncoderBlock_In(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock_In, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        return x

class DecoderBlock_In(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock_In, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        return x


class DecoderBlock_Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock_Out, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv_transpose1(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        return x
    

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.encoder1 = EncoderBlock_In(1, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.encoder4 = EncoderBlock(128, 256)
        self.encoder5 = EncoderBlock(256, 512)
        self.encoder6 = EncoderBlock_Out(512, 1024)

        self.featuremaps = nn.AdaptiveMaxPool2d((1,1))

        self.decoder6 = DecoderBlock_In(1024, 512)
        self.decoder5 = DecoderBlock(1024, 256)
        self.decoder4 = DecoderBlock(512, 128)
        self.decoder3 = DecoderBlock(256, 64)
        self.decoder2 = DecoderBlock(128, 32)
        self.decoder1 = DecoderBlock_Out(32, 2)

        self.attention2 = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1)
        self.attention3 = nn.MultiheadAttention(embed_dim=128, num_heads=1, dropout=0.1)
        self.attention4 = nn.MultiheadAttention(embed_dim=256, num_heads=1, dropout=0.1)
        self.attention5 = nn.MultiheadAttention(embed_dim=512, num_heads=1, dropout=0.1)
        self.attention6 = nn.MultiheadAttention(embed_dim=1024, num_heads=1, dropout=0.1)

    def apply_attention(self, feat1, feat2, attention_layer):
        feat1_flat = feat1.flatten(2).permute(2, 0, 1) 
        feat2_flat = feat2.flatten(2).permute(2, 0, 1)
        fused_feat, _ = attention_layer(feat1_flat, feat2_flat, feat2_flat)
        return fused_feat.permute(1, 2, 0).view_as(feat1)

    def forward(self, x):

        img_1 = x [:, 0:1, :, :]
        img_2 = x [:, 1:2, :, :]
        
        c11 = self.encoder1(img_1)
        c12 = self.encoder1(img_2)
        
        c21 = self.encoder2(c11)
        c22 = self.encoder2(c12)
        f2 = self.apply_attention(c21, c22, self.attention2)
        
        c31 = self.encoder3(c21)
        c32 = self.encoder3(c22)
        f3 = self.apply_attention(c31, c32, self.attention3)
        
        c41 = self.encoder4(c31)
        c42 = self.encoder4(c32)
        f4 = self.apply_attention(c41, c42, self.attention4)
        
        c51 = self.encoder5(c41)
        c52 = self.encoder5(c42)
        f5 = self.apply_attention(c51, c52, self.attention5)
        
        c61 = self.encoder6(c51)
        c62 = self.encoder6(c52)
        f6 = self.apply_attention(c61, c62, self.attention6)
        

        d6 = self.decoder6(f6)

        d5 = self.decoder5(torch.cat((d6,f5),1))

        d4 = self.decoder4(torch.cat((d5,f4),1))

        d3 = self.decoder3(torch.cat((d4,f3),1))

        d2 = self.decoder2(torch.cat((d3,f2),1))

        d1 = self.decoder1(d2)
        

        return d1