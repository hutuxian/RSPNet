import functools
import logging
from typing import *

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn

logger = logging.getLogger(__name__)


class Flatten(nn.Layer):
    def forward(self, x):
        return x.flatten(1)


class ConvFc(nn.Layer):
    """
    conv->relu->conv->downsample->linear

    """

    def __init__(self, feat_dim, moco_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv3D(feat_dim, feat_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3D(feat_dim, feat_dim, kernel_size, padding=padding)
        self.avg_pool = nn.AdaptiveAvgPool3D((1, 1, 1))
        self.linear = nn.Linear(feat_dim, moco_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.linear(out)
        return out


class ConvBnFc(nn.Layer):
    """
    conv->relu->conv->downsample->linear

    """

    def __init__(self, feat_dim, moco_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv3D(feat_dim, feat_dim, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3D(feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool3D((1, 1, 1))
        self.linear = nn.Linear(feat_dim, moco_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.linear(out)
        return out


class MultiTaskWrapper(nn.Layer):
    """
    This wrapper adds two independent projection layers (one for each pretext task) behind the backbone network.
    The projection layer type can be linear layer and mlp (as indicated in SimCLR).
    """
    def __init__(
            self,
            base_encoder,
            num_classes = 128,
            finetune = False,
            fc_type = 'linear',
            groups = 1,
    ):
        """
        :param base_encoder:
        :param num_classes:
        :param finetune:
        :param fc_type:
        :param groups:
        """
        super(MultiTaskWrapper, self).__init__()

        logger.info('Using MultiTask Wrapper')
        self.finetune = finetune
        self.moco_dim = num_classes
        self.num_classes = num_classes
        self.groups = groups
        self.fc_type = fc_type

        # logger.warning(f'{self.__class__} using groups: {groups}')

        self.encoder = base_encoder(num_classes=1)

        feat_dim = self._get_feat_dim(self.encoder)
        feat_dim //= groups

        if self.finetune:
            self.avg_pool = nn.AdaptiveAvgPool3D((1, 1, 1))
            self.fc = nn.Linear(feat_dim, num_classes)
        else:
            if fc_type == 'linear':
                self.fc1 = self._get_linear_fc(feat_dim, self.moco_dim)
                self.fc2 = self._get_linear_fc(feat_dim, self.moco_dim)
            elif fc_type == 'mlp':
                self.fc1 = self._get_mlp_fc(feat_dim, self.moco_dim)
                self.fc2 = self._get_mlp_fc(feat_dim, self.moco_dim)
            elif fc_type == 'conv':
                # A
                # self.fc1 = ConvFc(feat_dim, self.moco_dim, (1, 3, 3), (0, 1, 1))
                self.fc1 = ConvFc(feat_dim, self.moco_dim, (3, 3, 3), (1, 1, 1))
                # M
                # self.fc2 = ConvFc(feat_dim, self.moco_dim, (3, 1, 1), (1, 0, 0))
                self.fc2 = ConvFc(feat_dim, self.moco_dim, (3, 3, 3), (1, 1, 1))
            elif fc_type == 'convbn':
                self.fc1 = ConvBnFc(feat_dim, self.moco_dim, (3, 3, 3), (1, 1, 1))
                self.fc2 = ConvBnFc(feat_dim, self.moco_dim, (3, 3, 3), (1, 1, 1))
            elif fc_type == 'speednet':
                self.fc1 = self._get_linear_fc(feat_dim, self.moco_dim)
                self.fc2 = self._get_linear_fc(feat_dim, 1)  # use for speed binary classification like speednet

    def forward(self, x):
        feat = self.encoder.get_feature(x)
        print("feat: ", feat)

        if self.finetune:
            x3 = self.avg_pool(feat)
            x3 = x3.flatten(1)
            x3 = self.fc(x3)
            return x3
        else:
            if self.groups == 1:
                x1 = self.fc1(feat)
                x2 = self.fc2(feat)
            elif self.groups == 2:
                feat1, feat2 = feat.chunk(2, 1)
                x1 = self.fc1(feat1)
                x2 = self.fc2(feat2)
            else:
                raise Exception
            x1 = F.normalize(x1, axis=1)

            if self.fc_type == 'speednet':  # for speednet, it use sigmoid to ouput the probability that whether the clip is sped up 
                x2 = paddle.sigmoid(x2)
            else:
                x2 = F.normalize(x2, axis=1)
            return x1, x2

    @staticmethod
    def _get_linear_fc(feat_dim, moco_dim):
        return nn.Sequential(
            nn.AdaptiveAvgPool3D((1, 1, 1)),
            Flatten(),
            nn.Linear(feat_dim, moco_dim),
        )

    @staticmethod
    def _get_mlp_fc(feat_dim, moco_dim):
        return nn.Sequential(
            nn.AdaptiveAvgPool3D((1, 1, 1)),
            Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, moco_dim)
        )

    @staticmethod
    def _get_feat_dim(encoder):
        fc_names = ['fc', 'new_fc', 'classifier']
        feat_dim = 512
        for fc_name in fc_names:
            if hasattr(encoder, fc_name):
                feat_dim = getattr(encoder, fc_name).weight.shape[0]
                logger.warning('Found fc: %s with in_features: %d' % (fc_name, feat_dim))
                break
        return feat_dim
