import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DilatedConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DilatedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=2, dilation=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(GatedFusionBlock, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_texture, feat_geo):
        cat_feat = torch.cat([feat_texture, feat_geo], dim=1)
        alpha = self.gate_conv(cat_feat)
        fused_feat = feat_texture * alpha + feat_geo * (1 - alpha)
        return fused_feat


class GridBlock(nn.Module):
    def __init__(self, in_channels, out_channels, long_kernel=5):
        super(GridBlock, self).__init__()
        mid_channels = out_channels // 2
        self.branch_h = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, long_kernel), padding=(0, long_kernel // 2)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (long_kernel, 1), padding=(long_kernel // 2, 0)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_node = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        h = self.branch_h(x)
        v = self.branch_v(x)
        n = self.branch_node(x)
        combined = torch.cat([h, v, n], dim=1)
        out = self.fusion(combined)
        return out * self.ca(out)


class GridEncoder(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(GridEncoder, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)
        self.layer1 = nn.Sequential(
            GridBlock(base_channels, base_channels * 2),
            DilatedConv(base_channels * 2, base_channels * 2),
        )
        self.down2 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            GridBlock(base_channels * 2, base_channels * 4),
            DilatedConv(base_channels * 4, base_channels * 4),
        )
        self.down3 = nn.MaxPool2d(2)
        self.layer3 = nn.Sequential(
            GridBlock(base_channels * 4, base_channels * 8),
            DilatedConv(base_channels * 8, base_channels * 8),
        )
        self.down4 = nn.MaxPool2d(2)
        self.layer4 = nn.Sequential(
            GridBlock(base_channels * 8, base_channels * 16),
            DilatedConv(base_channels * 16, base_channels * 16),
        )

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.layer1(self.down1(x0))
        x2 = self.layer2(self.down2(x1))
        x3 = self.layer3(self.down3(x2))
        x4 = self.layer4(self.down4(x3))
        return [x0, x1, x2, x3, x4]


class GridDecoder(nn.Module):
    def __init__(self, out_channels, base_channels):
        super(GridDecoder, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(base_channels * 16 + base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(base_channels * 8 + base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        d4 = self.up1(x4)
        if d4.size() != x3.size():
            d4 = F.interpolate(d4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        d4 = self.conv_up1(torch.cat([d4, x3], dim=1))
        d3 = self.up2(d4)
        if d3.size() != x2.size():
            d3 = F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        d3 = self.conv_up2(torch.cat([d3, x2], dim=1))
        d2 = self.up3(d3)
        if d2.size() != x1.size():
            d2 = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.conv_up3(torch.cat([d2, x1], dim=1))
        d1 = self.up4(d2)
        if d1.size() != x0.size():
            d1 = F.interpolate(d1, size=x0.shape[2:], mode="bilinear", align_corners=True)
        d1 = self.conv_up4(torch.cat([d1, x0], dim=1))
        return self.tail(d1)


class GatedGridUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=32):
        super(GatedGridUNet, self).__init__()
        if in_channels == 4:
            self.texture_channels = 2
            self.geo_channels = 2
        else:
            self.texture_channels = 1
            self.geo_channels = 2

        self.encoder_texture = GridEncoder(in_channels=self.texture_channels, base_channels=base_channels)
        self.encoder_geo = GridEncoder(in_channels=self.geo_channels, base_channels=base_channels)
        self.gate0 = GatedFusionBlock(base_channels)
        self.gate1 = GatedFusionBlock(base_channels * 2)
        self.gate2 = GatedFusionBlock(base_channels * 4)
        self.gate3 = GatedFusionBlock(base_channels * 8)
        self.gate4 = GatedFusionBlock(base_channels * 16)
        self.decoder = GridDecoder(out_channels, base_channels=base_channels)

    def forward(self, x):
        tc, gc = self.texture_channels, self.geo_channels
        img_texture = x[:, 0:tc, :, :]
        img_geo = x[:, tc : tc + gc, :, :]
        feats_t = self.encoder_texture(img_texture)
        feats_g = self.encoder_geo(img_geo)
        fused_feats = [
            self.gate0(feats_t[0], feats_g[0]),
            self.gate1(feats_t[1], feats_g[1]),
            self.gate2(feats_t[2], feats_g[2]),
            self.gate3(feats_t[3], feats_g[3]),
            self.gate4(feats_t[4], feats_g[4]),
        ]
        return self.decoder(fused_feats)


class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        skel = torch.zeros_like(img)
        temp = img.clone()
        for _ in range(self.num_iter):
            eroded = self.soft_erode(temp)
            opened = self.soft_open(eroded)
            skel = skel + (temp - opened)
            temp = eroded
        return torch.sigmoid(skel * 5.0)


def soft_cldice_loss(pred, target, skel_extractor):
    skel_pred = skel_extractor(pred)
    skel_true = skel_extractor(target)
    tprec = (skel_pred * target).sum(dtype=torch.float32) / (skel_pred.sum(dtype=torch.float32) + 1e-5)
    tsens = (skel_true * pred).sum(dtype=torch.float32) / (skel_true.sum(dtype=torch.float32) + 1e-5)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-5)
    return 1.0 - cl_dice


class TopoLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, weight_cldice=0.2):
        super(TopoLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.skel_extractor = SoftSkeletonize(num_iter=5)
        self.alpha = alpha
        self.beta = beta
        self.weight_cldice = weight_cldice

    def forward(self, pred_logits, target):
        loss_bce = self.bce(pred_logits, target)
        pred_probs = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred_probs * target).sum(dtype=torch.float32)
        union = pred_probs.sum(dtype=torch.float32) + target.sum(dtype=torch.float32)
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        loss_dice = 1.0 - dice_score
        loss_cldice = soft_cldice_loss(pred_probs, target, self.skel_extractor)
        return (self.alpha * loss_bce) + (self.beta * loss_dice) + (self.weight_cldice * loss_cldice)
