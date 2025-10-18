import torch

def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"\rTraining: {((i+1)/len(loader))*100:.1f}%", end="")
    print()
    return total_loss / len(loader)


def eval_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
    return total_loss / len(loader)
