import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha, beta, temperature):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, output_student, output_teacher, target):
        soft_output_student = F.softmax(output_student.logits / self.temperature, dim=1)
        soft_output_teacher = F.softmax(output_teacher.logits / self.temperature, dim=1)
        first_objective = F.cross_entropy(
            soft_output_student,
            soft_output_teacher,
        )
        first_objective *= self.temperature**2
        second_objective = F.cross_entropy(output_student.logits, target)
        return first_objective * self.alpha + second_objective * self.beta
