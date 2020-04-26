import torch.optim as optim
import torch.nn as nn
import torch
from cnn_feature_extraction import CNN
from image_dataset import ShapeImageDataset

# edit this to reflect real dataset file
train_path = "train.p"
test_path = "test.p"

train_set = ShapeImageDataset(train_path)
test_set = ShapeImageDataset(test_path)

#Test loader has constant batch sizes, so we can define it directly
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)

def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
    return train_loader

def cnn_train(net, epochs, batch_size, lr):
    cnn_net = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_net.parameters(), lr=lr, momentum=0.9)

    trainloader = get_train_loader(batch_size)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = 'cnn_net.pt'
    torch.save(net.state_dict(), PATH)

    print('Finished Training')