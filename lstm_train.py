import torch.optim as optim
import torch.nn as nn
import torch
from torchvision import transforms, utils
from lstm_model import VideoLSTM
from features_dataset import FeaturesDataset, ToTensor

cnn_json_file = "data/cnn_video_results.json"
yolo_json_file = "data/results_valid_video_all.json"
video_file = "data/videos/images.txt"
labels_file = "data/videos/labels.txt"
dataset_size = 1000

# 64 features from CNN, 6 from YOLO
input_size = 70
# trying 100 for now, decrease if overfitting and increase if underfitting
hidden_size = 100
# 3 for one-hot classification, 4 for bounding box, 1 for confidence
output_size = 8
# trying 2 for now; might need more
hidden_layers = 2

full_dataset = FeaturesDataset(cnn_json_file, yolo_json_file, video_file, labels_file, transform=transforms.Compose([ToTensor()]))
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                               shuffle=False, num_workers=2)

epochs = 100

# TODO: this is currently a rudimentary implementation; no special handling of detection failure atm
# both outputs and targets are tuples (class one-hot, bbox, confidence)
# output is in this format already, but target will have to be reformatted before this function is called
def custom_loss(output, target):
    class_output, bbox_output, conf_output = output
    class_target, bbox_target, conf_target = target

    class_loss = nn.BCEWithLogitsLoss()(class_output, class_target)
    # try just MSE with bbox, but might need to switch to YOLO method
    bbox_loss = nn.MSELoss()(bbox_output, bbox_target)
    conf_loss = nn.MSELoss()(conf_output, conf_target)

    loss = class_loss + 5*bbox_loss + conf_loss

    return loss


if __name__ == "__main__":
    # GPU training if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VideoLSTM(input_size, hidden_size, hidden_layers, output_size)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs and targets; paths not needed for training
            inputs, targets = data['inputs'], data['targets']
            # reorganize targets to be compatible with outputs
            targets = [targets[:, :, :3], targets[:, :, 3:7], targets[:, :, 7:]]

            # change ByteTensor (default) to FloatTensor
            inputs = inputs.type('torch.FloatTensor').to(device)
            targets = [t.type('torch.LongTensor').to(device) for t in targets]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if device.type == 'cpu':
                outputs = [i.type("torch.FloatTensor") for i in outputs]
            else:
                outputs = [i.type("torch.cuda.FloatTensor") for i in outputs]

            loss = custom_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

        PATH = 'trained_models/lstm_' + str(epochs) + 'epochs.pt'
        torch.save(model.state_dict(), PATH)

        print('Finished Training')



