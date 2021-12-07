import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from nets.net import MNISTNet, transform
from nets.config import train_kwargs, test_kwargs, params, device
from nets.serving_server import serving_model
from nets.utils import CustomedMNISTDataset
from core.models import Training
from core.utils import version_weights

# FLAGS
flags = {"is_training": False, "training_id": -1, "train_dataset_updated": False}
training_messages = {"validation": "", "epoch": ""}

training_model = MNISTNet()
optimizer = optim.Adadelta(training_model.parameters(), lr=params.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=params.gamma)

def get_training_metrics(status):
    training_id = flags["training_id"]
    validation = training_messages["validation"]
    epoch = training_messages["epoch"]
    return Training(training_id=training_id, status=status, validation=validation, epoch=epoch) 
    

def train(params, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % params.log_interval == 0:
            training_messages["epoch"] = 'Epoch {} Training [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                            100. * batch_idx / len(train_loader), loss.item())

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    training_messages["validation"] = 'Epoch {} Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'\
                                        .format(epoch, test_loss, correct, len(test_loader.dataset),
                                        100. * correct / len(test_loader.dataset))

def publish_new_weights(new_model_path):
    # Trigger serving model to update weights
    serving_model.load_state_dict(torch.load(new_model_path))
    print("Serving model load new weights")


def do_train():
    train_dataset = CustomedMNISTDataset(data_path = "nets/datasets/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_dataset = CustomedMNISTDataset(data_path = "nets/datasets/validation", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,**test_kwargs)

    training_messages["validation"] = ""
    training_messages["epoch"] = ""

    for epoch in range(1, params.epochs + 1):
        test(training_model, device, test_loader, epoch)
        train(params, training_model, device, train_loader, optimizer, epoch)
        scheduler.step()
    test(training_model, device, test_loader, epoch+1)
    flags["is_training"] = False

    if params.save_model:
        new_model_path = "nets/weights/mnist_cnn.pt"
        torch.save(training_model.state_dict(), new_model_path)
        version_weights()
        publish_new_weights(new_model_path)

