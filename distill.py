import os
import torch
from tqdm import tqdm
from model import (
    get_pretrained_gpt2,
    get_untrained_distilgpt2,
)
from dataset import get_dataloaders


def distill(config):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    teacher_model = get_pretrained_gpt2()
    teacher_model.to(device)

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
        student_model = get_untrained_distilgpt2(config)
        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=config["training_parameters"]["learning_rate"],
        )
        epoch_start = 0
        val_loss_best = None

    student_model.to(device)

    train_loader, valid_loader, test_loader = get_dataloaders(config)

    criterion = config["training_parameters"]["loss"]
    criterion = criterion.to(device)

    for epoch in range(epoch_start, config["training_parameters"]["nb_epochs"]):
        print(
            f"""#########   Epoch {epoch + 1}/{config["training_parameters"]["nb_epochs"]}   #########"""
        )
        train_loss = train_one_epoch(
            teacher_model, student_model, train_loader, optimizer, criterion, device
        )
        val_loss = evaluate(
            teacher_model, student_model, valid_loader, criterion, device
        )
        print(
            f"epoch: {epoch} training loss: {train_loss:.3f} validation loss: {val_loss:.3f}\n"
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
    student_model.train()  # switch to train mode

    print("Training :")
    for batch_idx, (input, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        # take a batch
        for key in input:
            input[key] = input[key].to(device)
        target = target.to(device)
        output_teacher = teacher_model(**input)
        # forward pass
        output_student = student_model(**input)
        loss = loss_fn(output_student, output_teacher, target)
        loss_it.append(loss.item())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

    return sum(loss_it) / len(loss_it)


def evaluate(teacher_model, student_model, loader, loss_fn, device):
    loss_it = list()
    student_model.eval()  # switch to train mode

    print("Evaluation :")
    for batch_idx, (input, target) in tqdm(enumerate(loader), total=len(loader)):
        # take a batch
        for key in input:
            input[key] = input[key].to(device)
        target = target.to(device)
        # forward pass
        with torch.no_grad():
            output_teacher = teacher_model(**input)
            output_student = student_model(**input)
            loss = loss_fn(output_student, output_teacher, target)
        loss_it.append(loss.item())

    return sum(loss_it) / len(loss_it)


if __name__ == "__main__":
    import configue

    config = configue.load("config.yaml")

    distill(config)
