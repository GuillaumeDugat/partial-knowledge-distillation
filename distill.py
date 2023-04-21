import os
import torch
from tqdm import tqdm
from model import (
    get_pretrained_gpt2,
    get_pretrained_distilgpt2,
)
from dataset import get_dataloaders
from loss import DistillationLoss


def distill(config):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    criterion = config["training_parameters"]["loss"]
    criterion = criterion.to(device)

    if isinstance(criterion, DistillationLoss):
        teacher_model = get_pretrained_gpt2()
        teacher_model.to(device)
    else:
        teacher_model = None  # makes sure teacher model is not called in this mode

    if config["training_parameters"]["resume"]:
        checkpoint_last = torch.load(
            os.path.join(
                config["paths"]["checkpoints"],
                config["training_parameters"]["checkpoints_name_last"],
            )
        )
        checkpoint_best = torch.load(
            os.path.join(
                config["paths"]["checkpoints"],
                config["training_parameters"]["checkpoints_name_best"],
            )
        )
        student_model = checkpoint_last["model"]
        optimizer = checkpoint_last["optimizer"]
        epoch_start = checkpoint_last["epoch"]
        val_loss_best = checkpoint_best["val_loss"]
    else:
        student_model = get_pretrained_distilgpt2()
        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=config["training_parameters"]["learning_rate"],
        )
        epoch_start = 0
        val_loss_best = None

    student_model.to(device)

    train_loader, valid_loader, test_loader = get_dataloaders(config)

    for epoch in range(epoch_start, config["training_parameters"]["nb_epochs"]):
        print(
            f"""#########   Epoch {epoch + 1}/{config["training_parameters"]["nb_epochs"]}   #########"""
        )
        train_loss, train_first_objective, train_second_objective = train_one_epoch(
            teacher_model, student_model, train_loader, optimizer, criterion, device
        )
        val_loss, val_first_objective, val_second_objective = evaluate(
            teacher_model, student_model, valid_loader, criterion, device
        )
        print(
            f"epoch: {epoch} training loss: {train_loss}\t{train_first_objective}\t{train_second_objective} | validation loss: {val_loss}\t{val_first_objective}\t{val_second_objective}\n"
        )
        # Save model
        checkpoint = {
            "epoch": epoch,
            "model": student_model,
            "optimizer": optimizer,
            "val_loss": val_loss,
        }
        torch.save(
            checkpoint,
            os.path.join(
                config["paths"]["checkpoints"],
                config["training_parameters"]["checkpoints_name_last"],
            ),
        )
        if val_loss_best is None or val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(
                checkpoint,
                os.path.join(
                    config["paths"]["checkpoints"],
                    config["training_parameters"]["checkpoints_name_last"],
                ),
            )


def train_one_epoch(
    teacher_model, student_model, train_loader, optimizer, loss_fn, device
):
    loss_it = list()
    loss_first = list()
    loss_second = list()
    student_model.train()  # switch to train mode

    print("Training :")
    for batch_idx, (input, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        # take a batch
        for key in input:
            input[key] = input[key].to(device)
        target = target.to(device)
        # forward pass
        output_student = student_model(**input)
        if isinstance(loss_fn, DistillationLoss):
            output_teacher = teacher_model(**input)
            loss = loss_fn(output_student, output_teacher, target)

            loss_first.append(loss_fn.first_objective.item())
            loss_second.append(loss_fn.second_objective.item())
        else:
            loss = loss_fn(output_student, target)

        loss_it.append(loss.item())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        if batch_idx % 500 == 0:
            print(
                f"Train [{batch_idx}/{len(train_loader)}]\t{get_avg(loss_it)} \t {get_avg(loss_first)} \t {get_avg(loss_second)}"
            )

    print(
        f"Train [{batch_idx}/{len(train_loader)}]\t{get_avg(loss_it)} \t {get_avg(loss_first)} \t {get_avg(loss_second)}"
    )

    return (
        get_avg(loss_it),
        get_avg(loss_first),
        get_avg(loss_second),
    )


def get_avg(loss_list):
    if len(loss_list) == 0:
        return 0
    else:
        return sum(loss_list) / len(loss_list)


def evaluate(teacher_model, student_model, loader, loss_fn, device):
    loss_it = list()
    loss_first = list()
    loss_second = list()
    student_model.eval()  # switch to eval mode

    print("Evaluation :")
    for batch_idx, (input, target) in tqdm(enumerate(loader), total=len(loader)):
        # take a batch
        for key in input:
            input[key] = input[key].to(device)
        target = target.to(device)
        # forward pass
        with torch.no_grad():
            output_student = student_model(**input)
            if isinstance(loss_fn, DistillationLoss):
                output_teacher = teacher_model(**input)
                loss = loss_fn(output_student, output_teacher, target)

                loss_first.append(loss_fn.first_objective.item())
                loss_second.append(loss_fn.second_objective.item())
            else:
                loss = loss_fn(output_student, target)

        loss_it.append(loss.item())

        if batch_idx % 500 == 0:
            print(
                f"Evaluate [{batch_idx}/{len(loader)}]\t{get_avg(loss_it)} \t {get_avg(loss_first)} \t {get_avg(loss_second)}"
            )

    print(
        f"Evaluate [{batch_idx}/{len(loader)}]\t{get_avg(loss_it)} \t {get_avg(loss_first)} \t {get_avg(loss_second)}"
    )

    return (
        get_avg(loss_it),
        get_avg(loss_first),
        get_avg(loss_second),
    )


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")

    distill(config)
