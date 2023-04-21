import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha, beta, temperature):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.first_objective = 0
        self.second_objective = 0

    def forward(
        self, output_student, output_teacher, target
    ):  # remove the three special tokens of gpt2 trained on alpaca
        output_teacher_cropped = output_teacher.logits[:, :, :-3]
        soft_output_student = F.softmax(output_student.logits / self.temperature, dim=1)
        soft_output_teacher = F.softmax(
            output_teacher_cropped / self.temperature, dim=1
        )
        first_objective = F.cross_entropy(
            soft_output_student,
            soft_output_teacher,
        )
        first_objective *= self.temperature**2
        # remove last predicted word from logit as we have no ground truth for it
        second_objective = F.cross_entropy(output_student.logits[:, :-1], target)

        self.first_objective = first_objective
        self.second_objective = second_objective

        return first_objective * self.alpha + second_objective * self.beta


class ClassicLoss(torch.nn.Module):
    def __init__(self):
        super(ClassicLoss, self).__init__()

    def forward(self, output_student, target):
        return F.cross_entropy(output_student.logits[:, :-1], target)
