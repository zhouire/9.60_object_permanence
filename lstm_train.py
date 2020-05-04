import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms, utils
from lstm_model import VideoLSTM
from features_dataset import FeaturesDataset, ToTensor

cnn_json_file = "data/cnn_video_results.json"
yolo_json_file = "data/results_valid_video_all.json"
video_file = "data/videos/images.txt"
labels_file = "data/videos/labels.txt"
dataset_size = 1000

# 64 features from CNN, 8 from YOLO
input_size = 72
# trying 100 for now, decrease if overfitting and increase if underfitting
hidden_size = 100
# 3 for one-hot classification, 4 for bounding box
output_sizes = (3, 4)
# trying 2 for now; might need more
hidden_layers = 2

epochs = 2000

# TODO: this is currently a rudimentary implementation; no special handling of detection failure atm
# both outputs and targets are tuples (class one-hot, bbox, confidence)
# output is in this format already, but target will have to be reformatted before this function is called
def custom_loss(output, target):
    class_output, bbox_output = output
    class_target, bbox_target = target
    class_target = class_target.squeeze()
    bbox_target = bbox_target.squeeze()

    #class_loss = nn.BCEWithLogitsLoss()(class_output, class_target)
    # crossentropy loss takes one-hot input but index target
    class_loss = nn.CrossEntropyLoss()(class_output, class_target)
    # try just MSE with bbox, but might need to switch to YOLO method
    bbox_loss = nn.MSELoss()(bbox_output, bbox_target)

    loss = class_loss + 10*bbox_loss

    return loss, class_loss, bbox_loss


if __name__ == "__main__":
    # GPU training if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # data loading
    full_dataset = FeaturesDataset(cnn_json_file, yolo_json_file, video_file, labels_file,
                                   transform=transforms.Compose([ToTensor()]))
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=2)

    # make the model
    model = VideoLSTM(input_size, hidden_size, hidden_layers, output_sizes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_bbox_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs and targets; paths not needed for training
            inputs, targets = data['inputs'], data['targets']
            inputs = np.swapaxes(inputs, 0, 1)
            targets = np.swapaxes(targets, 0, 1)

            # reorganize targets to be compatible with outputs
            targets = [targets[:, :, :1], targets[:, :, 1:]]

            # change ByteTensor (default) to FloatTensor
            inputs = inputs.type('torch.FloatTensor').to(device)
            targets = [targets[0].type('torch.LongTensor').to(device),
                       targets[1].type('torch.FloatTensor').to(device)]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if device.type == 'cpu':
                outputs = [i.type("torch.FloatTensor") for i in outputs]
            else:
                outputs = [i.type("torch.cuda.FloatTensor") for i in outputs]

            loss, class_loss, bbox_loss = custom_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_class_loss += class_loss.item()
            running_bbox_loss += bbox_loss.item()

            '''
            # print statistics
            if i % 400 == 399:  # print every 400 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
            '''
        # print loss every epoch
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / 800))
        print('[%d] class loss: %.3f' % (epoch + 1, running_class_loss / 800))
        print('[%d] bbox loss: %.3f' % (epoch + 1, running_bbox_loss / 800))
        running_loss = 0.0

    PATH = 'trained_models/lstm_' + str(epochs) + 'epochs.pt'
    torch.save(model.state_dict(), PATH)

    print('Finished Training')



