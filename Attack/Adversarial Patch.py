import argparse
import os 
import random
import torch 
import numpy as np 
from models.resnet import resnet50
from models.inceptionv3 import inception_v3
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.autograd import Variable
import sys 
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
from utils import progress_bar, format_time, submatrix
from utils import circle_transform, init_patch_circle, init_patch_square, square_transform, set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--cuda', default= True,action='store_true', help='enables cuda')
parser.add_argument('--target', type=int, default=0, help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')
parser.add_argument('--max_count', type=int, default=500, help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ')
parser.add_argument('--train_size', type=int, default=100, help='Number of training images')
parser.add_argument('--test_size', type=int, default=100, help='Number of test images')
parser.add_argument('--image_size', type=int, default=299, help='the height / width of the input image to network')
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')
parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
parser.add_argument('--gpu', type=int, default=0)
opt = parser.parse_args()

try:  os.makedirs(opt.outf)
except OSError:  pass

set_seed(opt.manualSeed, opt.cuda, opt.gpu)

net = inception_v3(num_classes = 5, aux_logits=False).cuda()
net.load_state_dict(torch.load(os.path.join("./checkpoint", net.__class__.__name__ + '-best' + '.pth')))
net.eval()

target = opt.target # 目标类别
conf_target = opt.conf_target # 训练时：目标分类器对目标类别的置信度达到该值时停止攻击，默认值为 0.9。
max_count = opt.max_count # 寻找对抗样本的最大迭代次数
patch_type = opt.patch_type # 补丁的类型
patch_size = opt.patch_size # 补丁的尺寸
image_size = opt.image_size # 图像的尺寸
train_size = opt.train_size # 训练图像的数量
test_size = opt.test_size   # 测试图像的数量
plot_all = opt.plot_all     # 是否绘制所有成功的对抗图像

# 从val中确定训练集和验证集
idx = np.arange(250)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size: train_size + test_size]

mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]) # TODO: set the dataset's mean and std manually
normalize = transforms.Normalize(mean, std)     
min_in, max_in = 0, 1
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

transform = transforms.Compose([
            transforms.Resize(round(image_size*1.050)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root=os.path.join("./dataset", "val"), transform=transform), 
    batch_size=1, shuffle=False, num_workers=8,
    sampler=SubsetRandomSampler(training_idx)
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root=os.path.join("./dataset", "val"), transform=transform),
    batch_size=1, shuffle=False, num_workers=8,
    sampler=SubsetRandomSampler(test_idx)
)

def attack(x, patch, mask):
    net.eval()
    x_out = F.softmax(net(x), dim = 1)
    target_prob = x_out.data[0][target]
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    count = 0 
    while conf_target > target_prob:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(net(adv_x), dim = 1)
        adv_out_probs, adv_out_labels = adv_out.max(1)
        Loss = -adv_out[0][target]
        Loss.backward()
        adv_grad = adv_x.grad.clone()
        adv_x.grad.data.zero_()
        patch -= adv_grad 
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        out = F.softmax(net(adv_x), dim = 1)
        target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]
        #print(count, conf_target, target_prob, y_argmax_prob)  
        if count >= opt.max_count:  break
    return adv_x, mask, patch 

def train(epoch, patch, patch_shape):
    net.eval()
    success = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda();labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction = net(data)
        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]: continue
        total += 1
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
        adv_x, mask, patch = attack(data, patch, mask)
        adv_label = net(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        if adv_label == target:
            success += 1
            if plot_all == 1:
              # plot source image
              vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)   
              # plot adversarial image
              vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch 
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success/total))
    return patch

def test(epoch, patch, patch_shape):
    net.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(val_loader):
        if opt.cuda: data = data.cuda();labels = labels.cuda()
        data, labels = Variable(data), Variable(labels)
        prediction = net(data)
        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:    continue
        total += 1
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda: patch, mask = patch.cuda(), mask.cuda()
        patch, mask = Variable(patch), Variable(mask)
        adv_x = torch.mul((1-mask),data) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        adv_label = net(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        if adv_label == target:  success += 1
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch
        progress_bar(batch_idx, len(val_loader), "Test Success: {:.3f}".format(success/total))

if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        test(epoch, patch, patch_shape)
