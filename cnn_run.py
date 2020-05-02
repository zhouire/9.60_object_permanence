import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms, utils
from cnn_feature_extraction import CNN
from video_dataset import VideoCNNDataset, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = CNN()
net.load_state_dict(torch.load("trained_models/cnn_net_10epoch.pt", map_location=device))

test_images = "data/videos/allimages.txt"
test_set = VideoCNNDataset(test_images, transform=transforms.Compose([ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                           shuffle=False, num_workers=2)

total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        images, paths, labels = data["image"], data['path'], data['annotation']
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print(paths)

print('Accuracy of the network on video frames: %d %%' % (
    100 * correct / total))