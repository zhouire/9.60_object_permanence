# this code tests + runs the lstm on the test video dataset
# first implementation: 87% classification acc, 26% average iou (250 - 1000 epochs all same)
# second implementation: (MSE of sqrt values in bbox, bbox multiplier decreased to 5) 87% class, 18% iou (250 epochs)
# 3: (no more sqrt, multiplier 50) fantastically poor training
# 4: (multiplier 20, Adam optimizer (instead of SGD) 0.5 lr (instead of 0.1)) training not budging immediately, even worse
# 5: (multiplier 20, Adam optimizer w/ 0.1 lr) same problem
# 6: (multiplier 20, SGD w/ 0.2 lr) classification working better, but regression wont budge
# 7: (multiplier 10, SGD w/ 0.05 lr) classification 87, bbox 31, (500, 750, 1000 epochs)
# 8: (sgd 0.01 lr) classification 86, bbox 35 (250-1000 epochs)
# 9: (sgd, 0.001 lr) classification 87, bbox 35 (250-1000 epochs)
# 10: (sgd, 0.01 lr, 5x) classification 87, bbox 32 (250-1000 epochs)
# 11: (no relu, Adam optim w/ 0.01 lr) classification 87, bbox 36 (250 epochs)
# 12: (no relu, Adam optim w/0.001 lr) classification 87, bbox 39
# 13: (no relu, Adam optim w/ 0.003 lr) classification 87, bbox 43 (looks like 1000 epochs was not enough to complete train)
# 14: (RMSE instead of MSE) classification 86, bbox 45
# 15: (longrun, Adam 0.001) classification 87, bbox 48 (running for more than 1000 epochs is unhelpful/slightly detrimental)



import torch.optim as optim
import torch.nn as nn
import torch
import json
import numpy as np
from torchvision import transforms, utils
from lstm_model import VideoLSTM
from features_dataset import FeaturesDataset, ToTensor
from iou import iou


def test(model, test_loader, device, savefile):
    json_output = []

    total = 0
    class_correct = 0
    bbox_iou = 0
    with torch.no_grad():
        for data in test_loader:
            # ASSUMPTION: BATCH_SIZE = 1
            inputs, paths, targets = data['inputs'], data['paths'], data['targets']
            inputs = np.swapaxes(inputs, 0, 1)
            targets = targets.numpy().squeeze()
            seq_len = inputs.size(0)

            # reorganize targets to be compatible with outputs
            class_target = targets[:, :1].squeeze()
            bbox_target = targets[:, 1:].tolist()

            # change ByteTensor (default) to FloatTensor
            inputs = inputs.type('torch.FloatTensor').to(device)

            # get outputs in a usable form
            class_output, bbox_output = model(inputs)
            class_output = class_output.data.detach().cpu().numpy().tolist()
            bbox_output = bbox_output.data.detach().cpu().numpy().tolist()

            #_, class_pred = torch.max(class_output.data, 1)
            class_pred = np.argmax(class_output, 1).tolist()
            total += seq_len
            class_correct += sum(class_pred == class_target)
            ious = [iou(bbox_output[i], bbox_target[i], pixel=False, has_class=False) for i in range(seq_len)]
            bbox_iou += sum(ious)

            json_output.append({"video": paths,
                                "class_pred": class_pred,
                                "bbox_pred": bbox_output,
                                "class_target": class_target.tolist(),
                                "bbox_target": bbox_target,
                                "iou": ious})

    with open(savefile, 'w') as outfile:
        json.dump(json_output, outfile)

    print('Classification test accuracy of the network: ' + str(100 * class_correct / total))
    print('Bbox test avg iou of the network: ' + str(100 * bbox_iou / total))

'''
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
'''



if __name__ == "__main__":
    # 64 features from CNN, 8 from YOLO
    input_size = 72
    # trying 100 for now, decrease if overfitting and increase if underfitting
    hidden_size = 100
    # 3 for one-hot classification, 4 for bounding box
    output_sizes = (3, 4)
    # trying 2 for now; might need more
    hidden_layers = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VideoLSTM(input_size, hidden_size, hidden_layers, output_sizes)
    model.load_state_dict(torch.load("trained_models/lstm_longrun_1000epochs.pt", map_location=device))
    model.to(device)

    cnn_json = "data/cnn_testvideo_results.json"
    yolo_json = "data/yolo_testvideo_results.json"
    video_file = "data/videos_test/images.txt"
    labels_file = "data/videos_test/labels.txt"

    test_set = FeaturesDataset(cnn_json, yolo_json, video_file, labels_file,
                               transform=transforms.Compose([ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                               shuffle=False, num_workers=2)

    test(model, testloader, device, 'data/lstm_testvideo_results.json')
