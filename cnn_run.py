# this code runs the CNN on video frames.
# Accuracy of the CNN for labeling video frame: 85% (this makes sense bc of occlusions)

import torch.optim as optim
import torch.nn as nn
import torch
import json
from torchvision import transforms, utils
from cnn_feature_extraction import CNN
from video_dataset import VideoCNNDataset, ToTensor


def test(net, test_loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, paths, labels = data["image"], data['path'], data['annotation']

            images = images.type('torch.FloatTensor').to(device)
            labels = labels.squeeze(1).type('torch.LongTensor').to(device)

            outputs = net(images)
            outputs = outputs.type("torch.FloatTensor").to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on video frames: %d %%' % (
        100 * correct / total))


def get_features(net, test_loader, device, savefile):
    json_output = []

    with torch.no_grad():
        for data in test_loader:
            images, paths, labels = data["image"], data['path'], data['annotation']

            images = images.type('torch.FloatTensor').to(device)
            labels = labels.squeeze(1).type('torch.LongTensor').to(device)

            outputs = net.get_feature_vec(images)

            features = outputs.detach().cpu().numpy().tolist()
            json_output.append({"image": paths, "features": features})

    with open(savefile, 'w') as outfile:
        json.dump(json_output, outfile)



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    cnn = CNN()
    cnn.load_state_dict(torch.load("trained_models/cnn_net_10epoch.pt", map_location=device))
    cnn.to(device)

    test_images = "data/videos_test/allimages.txt"
    test_set = VideoCNNDataset(test_images, transform=transforms.Compose([ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                               shuffle=False, num_workers=2)

    get_features(cnn, testloader, device, 'data/cnn_testvideo_results.json')

#test(cnn, testloader, device)