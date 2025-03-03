import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from models.resnet import resnet50 
from models.inceptionv3 import inception_v3
import matplotlib.pyplot as plt 
from earlystopping import EarlyStopping
import numpy as np
import time 
from torch.optim.lr_scheduler import CosineAnnealingLR

# resnet resize 299 [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# inceptionv3  resize 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
def process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(299, scale = (0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees = 45),
            transforms.RandomGrayscale(p=0.2),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose(
            [transforms.Resize((299, 299)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set1", "flower_data1")  # flower data set path
    image_path = '../dataset/'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset) # TODO: 样本数量

    # {'daisy':0, 'dandelion':1, 'roses':2}
    flower_list = train_dataset.class_to_idx # 类别名称到索引的映射关系
    cla_dict = dict((val, key) for key, val in flower_list.items()) # 反转
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 50
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = resnet50(num_classes = 5).to(device)
    loss = nn.CrossEntropyLoss()
    epoch = 100 
    lr = 1e-4
    return  net, loss, train_loader, val_loader, device, epoch, lr  

def train(net, loss, train_dataloader, valid_dataloader, device, num_epoch, lr, optim='adam',scheduler_type='Cosine', init = True, checkpoint = None):
    def init_xavier(m):
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    if init:
        net.apply(init_xavier)
        
    print('training on:', device)
    net.to(device)
    # 优化器选择
    if optim == 'sgd': optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam': optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW': optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    # elif optim == 'ranger':
    #     optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=lr,
    #                        weight_decay=0)
    if scheduler_type == 'Cosine':
         lr_min = 0
         scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    # 用来保存每个epoch的Loss和acc以便最后画图
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    best_acc = 0.0
    path = '../checkpoint/'
    if checkpoint: 
        best_acc = eval(path, net, valid_dataloader, loss)
        print(f"======>包含模型权重，当前的最好的验证级准确率:[{best_acc / len(valid_dataloader)}]<======")
    
    # path = ".\\checkpoint\\" + net.__class__.__name__ + '-latest'+ '.pth'
    early_stopping = EarlyStopping(verbose=True)
    # 训练
    for epoch in range(num_epoch):
        print("\t \t \t  ————————————第 {} 轮训练开始————————————".format(epoch + 1))
        time.sleep(1)
        # 训练开始
        net.train()
        train_acc = 0
        train_loss = 0
        for batch in tqdm(train_dataloader, desc='train'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            Loss = loss(output, targets)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            train_acc += num_correct / imgs.shape[0]
            train_loss += Loss.item()

        scheduler.step()
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(train_loss / len(train_dataloader))

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in tqdm(valid_dataloader,desc='valid'):
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                eval_acc += num_correct / imgs.shape[0]

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_path = path + net.__class__.__name__ + '-best' + '.pth'
                os.makedirs(os.path.dirname(best_path), exist_ok=True)  
                torch.save(net.state_dict(), best_path)

            eval_losses.append(eval_loss / len(valid_dataloader))
            eval_acces.append(eval_acc / len(valid_dataloader))
            print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(epoch+1, train_loss / len(train_dataloader), train_acc / len(train_dataloader),eval_loss / len(valid_dataloader), eval_acc / len(valid_dataloader)))
            early_stopping(eval_loss, net)
            if early_stopping.early_stop:
                print("Early stopping!")
                break  
    return train_losses, train_acces, eval_losses, eval_acces

def show_acces(train_losses, train_acces, valid_losses ,valid_acces, num_epoch):#对准确率和loss画图显得直观
    plt.plot(1 + np.arange(len(train_losses)), train_losses, linewidth=1.5, linestyle='dashed', label='train_losses')
    plt.plot(1 + np.arange(len(train_acces)), train_acces, linewidth=1.5, linestyle='dashed', label='train_acces')
    plt.plot(1 + np.arange(len(valid_losses)), valid_losses, linewidth=1.5, linestyle='dashed', label='valid_loss')
    plt.plot(1 + np.arange(len(valid_acces)), valid_acces, linewidth=1.5, linestyle='dashed', label='valid_acces')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch, 1))
    plt.legend()
    plt.show()

def eval(path, net, val_loader, loss):
    net.load_state_dict(torch.load(os.path.join(path, net.__class__.__name__ + '-best' + '.pth')))
    eval_acc = 0
    eval_loss = 0
    net.eval()
    with torch.no_grad():
            for imgs, targets in tqdm(val_loader,desc='valid'):
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                eval_acc += num_correct / imgs.shape[0]
            return eval_acc


if __name__ == '__main__':
    path = "../checkpoint"
    net, loss, train_loader, val_loader, device, epoch, lr  = process()
    net.load_state_dict(torch.load(os.path.join(path, net.__class__.__name__ + '-best' + '.pth')))
    train_losses, train_acces, eval_losses, eval_acces = train(
        net, loss, train_loader, val_loader, device, epoch, lr, checkpoint=True
    )
    # show_acces(train_losses, train_acces,eval_losses ,eval_acces, num_epoch=1)

