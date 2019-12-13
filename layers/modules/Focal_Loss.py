import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable



class FocalLoss(nn.Module):

    r"""

        This criterion is a implemenation of Focal Loss, which is proposed in

        Focal Loss for Dense Object Detection.



            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])



        The losses are averaged across observations for each minibatch.



        Args:

            alpha(1D Tensor, Variable) : the scalar factor for this criterion

            gamma(float, double) : gamma > 0; reduces the relative loss for well-clasified examples (p > .5),

                                   putting more focus on hard, misclassified examples

            size_average(bool): By default, the losses are averaged over observations for each minibatch.

                                However, if the field size_average is set to False, the losses are

                                instead summed for each minibatch.





    """

    def __init__(self, alpha, gamma=2, class_num=5,size_average=False):

        super(FocalLoss, self).__init__()

        if alpha is None:

            self.alpha = Variable(torch.ones(class_num, 1))

        else:

            if isinstance(alpha, Variable):

                self.alpha = alpha

            else:

                self.alpha = Variable(alpha)



        self.gamma = gamma



        #self.class_num = class_num

        self.size_average = size_average



    def forward(self, inputs, targets):
        n, c, h, w = input.size()

        target = target.long()
        inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.contiguous().view(-1)

        N = inputs.size(0)
        C = inputs.size(1)

        number_0 = torch.sum(target == 0).item()
        number_1 = torch.sum(target == 1).item()
        number_2 = torch.sum(target == 2).item()


        frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4, number_5), dtype=torch.float32)
        frequency = frequency.numpy()
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

