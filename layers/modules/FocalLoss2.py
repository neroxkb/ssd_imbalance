# -*- coding: utf-8 -*-

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from layers.box_utils import match, log_sum_exp

from data import coco as cfg
import numpy as np
np.set_printoptions(threshold=np.inf)
# I do not fully understand this part, It completely based on https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py


class FocalLoss2(nn.Module):
    """SSD Weighted Loss Function

    Focal Loss for Dense Object Detection.



        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])



    The losses are averaged across observations for each minibatch.

    Args:

        alpha(1D Tensor, Variable) : the scalar factor for this criterion

        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),

                                putting more focus on hard, misclassiﬁed examples

        size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.

                            However, if the field size_average is set to False, the losses are

                            instead summed for each minibatch.

    """

    def __init__(self, num_classes, overlap_thresh,
                 bkg_label,  neg_pos,alpha,gamma,
                 use_gpu=True):

        super(FocalLoss2, self).__init__()

        self.use_gpu = use_gpu

        self.num_classes = num_classes

        self.background_label = bkg_label

        self.negpos_ratio = neg_pos

        self.threshold = overlap_thresh

        #self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD

        self.variance = cfg['variance']

        #self.priors = priors

        self.alpha = alpha

        self.gamma = gamma

    def forward(self, predictions, targets):

        """Multibox Loss

        Args:

            predictions (tuple): A tuple containing loc preds, conf preds,

            and prior boxes from SSD net.

                conf shape: torch.size(batch_size,num_priors,num_classes)

                loc shape: torch.size(batch_size,num_priors,4)

                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,

                shape: [batch_size,num_objs,5] (last idx is the label).

        """

        loc_data, conf_data, priors = predictions

        num = loc_data.size(0)

        #priors = self.priors

        priors = priors[:loc_data.size(1), :]

        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes

        loc_t = torch.Tensor(num, num_priors, 4)

        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :-1].data

            labels = targets[idx][:, -1].data

            defaults = priors.data

            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()

            conf_t = conf_t.cuda()

        # wrap targets

        loc_t = Variable(loc_t, requires_grad=False)

        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)

        # Shape: [batch,num_priors,4]

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)

        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        loss_l /= num_pos.data.sum()
        N=num_pos.data.sum()
        # Confidence Loss (Focal loss)

        # Shape: [batch,num_priors,1]
        #print("conf_data.view(-1, self.num_classes)", conf_data.view(-1, self.num_classes))
        loss_c = self.focal_loss(conf_data.view(-1, self.num_classes), conf_t.view(-1, 1))
        print("loss_l",loss_l)
        print("loss_c",loss_c)
        return loss_l, loss_c

    def focal_loss(self, inputs, target):

        #n, c, h, w = input.size()

        #target = target.long()
        #inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        #target = target.contiguous().view(-1)

        N = inputs.size(0)
        C = inputs.size(1)

        number_0 = torch.sum(target == 0).item()
        number_1 = torch.sum(target == 1).item()
        number_2 = torch.sum(target == 2).item()


        frequency = torch.tensor((number_0, number_1, number_2), dtype=torch.float32)
        frequency = frequency.data.numpy()
        classWeights = compute_class_weights(frequency)

        weights = torch.from_numpy(classWeights).float()
        weights = weights[target.view(-1)]

        gamma = 2

        P = F.softmax(inputs, dim=1)  # shape [num_samples,num_classes]

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # shape [num_samples,num_classes]  one-hot encoding

        probs = (P * class_mask).sum(1).view(-1, 1)  # shape [num_samples,]
        log_p = probs.log()

        print('in calculating batch_loss', weights.shape, probs.shape, log_p.shape)

        # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
        batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

        print(batch_loss.shape)

        loss = batch_loss.mean()
        return loss