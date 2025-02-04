import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################
# v1 正则化
#################################################################
# class ReasoningLoss(nn.Module):
#     def __init__(self):
#         super(ReasoningLoss, self).__init__()


#     def forward(self, bn_features, bn_features2):
#         new_bn_features2 = torch.zeros(bn_features.size()).to(bn_features.device)
#         for i in range(int(bn_features2.size(0) / 4)):
#             new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
#         loss = torch.norm((bn_features - new_bn_features2), p=2)
#         return loss

#################################################################
# v2 标签蒸馏
#################################################################
# class KLDiv(nn.Module):
#     def __init__(self, temp=1.0):
#         super(KLDiv, self).__init__()
#         self.temp = temp

#     def forward(self, student_preds, teacher_preds, **kwargs):
#         soft_student_outputs = F.log_softmax(student_preds / self.temp, dim=1)
#         soft_teacher_outputs = F.softmax(teacher_preds / self.temp, dim=1)
#         kd_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction="none").sum(1).mean()
#         kd_loss *= self.temp**2
#         return kd_loss


# class ReasoningLoss(nn.Module):
#     def __init__(self):
#         super(ReasoningLoss, self).__init__()

#     def forward(self, student_preds, teacher_preds):
#         repeat_teacher_preds = teacher_preds.repeat_interleave(4, dim=0).clone().detach()
#         loss = KLDiv()(student_preds, repeat_teacher_preds)
#         return loss


#################################################################
# v2 标签蒸馏
#################################################################
class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def forward(self, student_features, teacher_features):
        repeat_teacher_features = teacher_features.repeat_interleave(4, dim=0).clone().detach()
        loss = 0
        loss += torch.norm((student_features), p=2)
        loss += 0.1 * torch.norm((student_features - repeat_teacher_features), p=2)
        return loss
