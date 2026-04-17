
import torch
import torch.nn.functional as F
import mlflow


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn,
               accuracy_fn,
               device: str = "cuda",
               scaler=None,
               scheduler=None):

    model.train()

    train_loss, train_acc = 0.0, 0.0
    num_batches = 0
    

    for X, y in dataloader:

        X = X.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_fn(outputs, y)
            
        if torch.isnan(outputs).any():
            print("NaNs in outputs")
            continue

        loss.backward()
        optimizer.step()

        if scheduler and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()

        preds = torch.argmax(outputs, dim=1)

        acc = accuracy_fn(preds, y)

        train_loss += loss.item()
        train_acc += acc.item()
        num_batches += 1

    return train_loss / num_batches, train_acc / num_batches


def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn,
                    accuracy_fn,
                    device: str = "cuda",):
    model.eval()

    total_loss, total_acc = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).long()

            outputs = model(X)
            loss = loss_fn(outputs, y)
                
            preds = torch.argmax(outputs, dim=1)

            acc = accuracy_fn(preds, y)

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

    # Average metrics for the epoch
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc


from tqdm.auto import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: callable,
          accuracy_fn: callable,
          scaler=None,
          scheduler=None,
          device: str = "cuda",
          epochs: int = 5):

    model.to(device)

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            scheduler=scheduler
        )

        test_loss, test_acc = validation_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("dice_score", test_acc, step=epoch)
        mlflow.log_metric("lr", current_lr, step=epoch)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results