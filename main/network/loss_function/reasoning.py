import torch
import torch.nn as nn
import torch.nn.functional as F

# class ReasoningLoss(nn.Module):
#     def __init__(self):
#         super(ReasoningLoss, self).__init__()


#     def forward(self, bn_features, bn_features2):
#         new_bn_features2 = torch.zeros(bn_features.size()).to(bn_features.device)
#         for i in range(int(bn_features2.size(0) / 4)):
#             new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
#         loss = torch.norm((bn_features - new_bn_features2), p=2)
#         return loss


class KLDiv(nn.Module):
    def __init__(self, temp=1.0):
        super(KLDiv, self).__init__()
        self.temp = temp

    def forward(self, student_preds, teacher_preds, **kwargs):
        soft_student_outputs = F.log_softmax(student_preds / self.temp, dim=1)
        soft_teacher_outputs = F.softmax(teacher_preds / self.temp, dim=1)
        kd_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction="none").sum(1).mean()
        kd_loss *= self.temp**2
        return kd_loss


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def forward(self, student_preds, teacher_preds):
        new_teacher_preds = torch.zeros(student_preds.size()).to(student_preds.device)
        for i in range(int(teacher_preds.size(0))):
            new_teacher_preds[i * 4 : (i + 1) * 4] = teacher_preds[i]
        loss = KLDiv()(student_preds, new_teacher_preds)
        return loss
