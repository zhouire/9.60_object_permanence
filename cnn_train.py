import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms, utils
from cnn_feature_extraction import CNN
from image_dataset import ShapeImageDataset, ToTensor

# edit this to reflect real dataset file
train_images = "data/pretrain/images.txt"
train_labels = "data/pretrain/labels.txt"
#test_path = "debug_images_test.p"

train_set = ShapeImageDataset(train_images, train_labels, yolo=False, transform=transforms.Compose([ToTensor()]))
#test_set = ShapeImageDataset(test_path, transform=transforms.Compose([ToTensor()]))

#Test loader has constant batch sizes, so we can define it directly
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

epochs = 10
batch_size = 64
lr = 0.001

def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
    return train_loader

if __name__ == "__main__":
    # GPU training if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = CNN()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    trainloader = get_train_loader(batch_size)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['annotation']
            # change ByteTensor (default) to FloatTensor
            inputs = inputs.type('torch.FloatTensor').to(device)
            labels = labels.squeeze(1).type('torch.LongTensor').to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if device.type == 'cpu':
                outputs = outputs.type("torch.FloatTensor")
            else:
                outputs = outputs.type("torch.cuda.FloatTensor")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    PATH = 'trained_models/cnn_net_10epoch.pt'
    torch.save(net.state_dict(), PATH)

    print('Finished Training')
