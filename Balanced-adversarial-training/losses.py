import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum

class DistanceMetric(Enum):
    """
    Metrics for computing the loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    def __init__(self, distance_metric, margin=1.0):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, anchor, positive, negative, mask=None):
        pos_dist = self.distance_metric(anchor, positive)
        neg_dist = self.distance_metric(anchor, negative)
        loss = F.relu(pos_dist + (self.margin - neg_dist) * mask)
        return loss.mean()


class PairwiseLoss(nn.Module):
    def __init__(self, distance_metric, margin=1.0):
        super(PairwiseLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, anchor, positive, negative, positive_mask, negative_mask):
        pos_dist = self.distance_metric(anchor, positive)
        neg_dist = self.distance_metric(anchor, negative)

        pos_loss = (pos_dist * positive_mask)
        neg_loss = F.relu(self.margin - neg_dist) * negative_mask

        pos_loss = pos_loss.sum() / positive_mask.sum()
        neg_loss = neg_loss.sum() / negative_mask.sum()
        return pos_loss, neg_loss