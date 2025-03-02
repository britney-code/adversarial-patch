import os
import sys
import time
import torch
import numpy as np
import math 
from scipy.ndimage import rotate
import random
import shutil
# _, term_width = os.popen('stty size', 'r').read().split()
_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    # 当前进度条的长度[已完成部分]
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    # 剩余进度条的长度[未完成部分]
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]

class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi)) # 初始化一个圆圈patch
    patch = np.zeros((1, 3, radius*2, radius*2)) # 根据圆的面积公式计算半径  
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius] # y = (74,1) 74行一列 x = (1, 74) 74列1行
        index = x**2 + y**2 <= radius**2 
        # 生成坐标网格后，判断每个点到圆心的距离是否小于半径 小于就是在圆内
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        # 将圆形区域内的值设置为随机值 
        idx = np.flatnonzero((a == 0).all((1))) # 删除全为0的行
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1) # 删除全为0的列
    return patch, patch.shape

def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
    # get shape
    m_size = patch_shape[-1]
    for i in range(x.shape[0]):
        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0
    return x, mask, patch.shape

def init_patch_square(data_shape, h_min, h_max, w_min, w_max):
    n, c, h, w = data_shape
    assert 0 <= h_min < h_max <= h, "Invalid height range"
    assert 0 <= w_min < w_max <= w, "Invalid width range"
    # get dummy image  patch, mask shape [1, 3, 299, 299]
    patch = torch.zeros(data_shape)
    mask = torch.zeros(data_shape)
    # 将区域opt.x_min, opt.x_max, opt.y_min, opt.y_max 方形范围设置为1
    mask[:, :, h_min : h_max, w_min : w_max] = 1
    # _patch = np.random.uniform(0.0, 1.0, (n, c, h_max - h_min, w_max - w_min))
    # _patch = torch.from_numpy(_patch)
    _patch = torch.rand(n, c, h_max - h_min, w_max - w_min)
    patch[:, :, h_min: h_max, w_min: w_max] = _patch
    return patch, mask

def square_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
    # get shape
    m_size = patch_shape[-1]
    for i in range(x.shape[0]):
        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]): patch[i][j] = np.rot90(patch[i][j], rot)
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    return x, mask

def set_seed(seed, cuda, device):
    if seed is None: 
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed_all(seed)  # TODO: 为所有的CUDA设备设置随机种子
        torch.cuda.set_device(device)
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

