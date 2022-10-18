import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm


import BasketballActionDataset from BasketballActionDataset

# Model Constants
WEIGHTS = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
LAYERS = ['layer3', 'layer4', 'fc']
NUM_CLASSES = 10
CLASSES = ['block', 'pass', 'run', 'dribble', 'shoot', 'ball in hand', 'defense', 'pick', 'no_action', 'walk'] 
BATCH_SIZE = 8
N_TOTAL = 49901
TEST_N = 4990
VAL_N = 9980

LR = 0.0001
START_EPOCH = 1
NUM_EPOCHS = 25

class BasketballActionRecognitionModel()
    def __init__(self):

        # Import a pretrained video action recognition model (R(2+1)D network)
        #  - Using the pretrained weights from the kinetics-400 dataset
        self.model = torchvision.models.video.r2plus1d_18(weights=WEIGHTS, progress=True)

        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            for layer in LAYERS:
                if layer in name:
                    param.requires_grad = True

        num_features = self.model.fc.in_features
        print(f'Number of Features: {num_features}')

        self.model.fc = torch.nn.Linear(num_features, NUM_CLASSES, bias=True)
        print(self.model)

        model_update_params = []
        print('Parameters to Update: ')
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(f'\t{name}')
                model_update_params.append(param)

        dataset = BasketballActionDataset('/dataset/annotation_dict.json')

        train_subset, test_subset = random_split(dataset, [N_TOTAL-TEST_N, TEST_N], generator=torch.Generator().manual_seed(1))

        train_subset, val_subset = random_split(train_subset, [N_TOTAL-TEST_N-VAL_N, VAL_N], generator=torch.Generator().manual_seed(1))

        train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=BATCH_SIZE)
        val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset=test_subset, shuffle=False, batch_size=BATCH_SIZE)

        self.dataloaders = {'train': train_loader, 'val': val_loader}

        self.optimizer_ft = torch.optim.Adam(params_to_update, lr=LR)
        self.criterion = nn.CrossEntropyLoss()


        # Train and evaluate
        train_l_hist, val_l_hist, train_acc_hist, val_acc_hist, train_f1_score, val_f1_score, plot_epoch = train_model()

    print("Best Validation Loss: ", min(val_loss_history), "Epoch: ", val_loss_history.index(min(val_loss_history)))
    print("Best Training Loss: ", min(train_loss_history), "Epoch: ", train_loss_history.index(min(train_loss_history)))

    # Plot Final Curve
    plot_curves(
        args.base_model_name,
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        train_f1_score,
        val_f1_score,
        plot_epoch
    )

    # Read History
    read_history(args.history_path)

    # Check Accuracy with Test Set
    check_accuracy(test_loader, model)

    def train_model(self):

    # Initializes Session History in the history file
    init_session_history(args)
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    train_f1_score = []
    val_f1_score = []
    plot_epoch = []

    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_acc = 0.0

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
                train_pred_classes = []
                train_ground_truths = []
            else:
                self.model.eval()  # Set model to evaluate mode
                val_pred_classes = []
                val_ground_truths = []

            running_loss = 0.0
            running_corrects = 0
            train_n_total = 1

            pbar = tqdm(self.dataloaders[phase])
            # Iterate over data.
            for sample in pbar:
                inputs = sample["video"]
                labels = sample["action"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, torch.max(labels, 1)[1])

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        train_pred_classes.extend(preds.detach().cpu().numpy())
                        train_ground_truths.extend(torch.max(labels, 1)[1].detach().cpu().numpy())
                    else:
                        val_pred_classes.extend(preds.detach().cpu().numpy())
                        val_ground_truths.extend(torch.max(labels, 1)[1].detach().cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

                pbar.set_description('Phase: {} || Epoch: {} || Loss {:.5f} '.format(phase, epoch, running_loss / train_n_total))
                train_n_total += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Calculate elapsed time
            time_elapsed = time.time() - since
            print(phase, ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # For Checkpointing and Confusion Matrix
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_pred_classes = np.asarray(val_pred_classes)
                val_ground_truths = np.asarray(val_ground_truths)
                val_accuracy, val_f1, val_precision, val_recall = get_acc_f1_precision_recall(
                    val_pred_classes, val_ground_truths
                )
                val_f1_score.append(val_f1)
                val_confusion_matrix = np.array_str(confusion_matrix(val_ground_truths, val_pred_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
                print('Epoch: {} || Val_Acc: {} || Val_Loss: {}'.format(
                    epoch, val_accuracy, epoch_loss
                ))
                print(f'val: \n{val_confusion_matrix}')

                # Deep Copy Model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # set current loss to val loss for write history
                val_loss = epoch_loss

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_pred_classes = np.asarray(train_pred_classes)
                train_ground_truths = np.asarray(train_ground_truths)
                train_accuracy, train_f1, train_precision, train_recall = get_acc_f1_precision_recall(
                    train_pred_classes, train_ground_truths
                )
                train_f1_score.append(train_f1)
                train_confusion_matrix = np.array_str(confusion_matrix(train_ground_truths, train_pred_classes, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
                print('Epoch: {} || Train_Acc: {} || Train_Loss: {}'.format(
                    epoch, train_accuracy, epoch_loss
                ))
                print(f'train: \n{train_confusion_matrix}')
                plot_epoch.append(epoch)

                # set current loss to train loss for write history
                train_loss = epoch_loss

        # Save Weights
        model_name = save_weights(self.model, epoch, self.optimizer)

        # Write History after train and validation phase
        write_history(
            args.history_path,
            model_name,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            train_f1,
            val_f1,
            train_precision,
            val_precision,
            train_recall,
            val_recall,
            train_confusion_matrix,
            val_confusion_matrix
        )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    self.model.load_state_dict(best_model_wts)
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_score, val_f1_score, plot_epoch






def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    self.model.eval()

    with torch.no_grad():
        i = args.batch_size

        pbar = tqdm(loader)
        for sample in pbar:
            x = sample["video"].to(device=device)
            y = sample["action"].to(device=device)

            scores = self.model(x)
            print(scores)
            predictions = scores.argmax (1)
            y = y.argmax (1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            pbar.set_description('Progress: {}'.format(i/args.test_n))
            i += args.batch_size

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    self.model.train()

        