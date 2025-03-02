import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image
    image_path = "../Attack/logs/xxx"
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('../class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
     print(e)
     exit(-1)

    # create model
    model = vgg(num_classes=4)
    # load model weights
    model_weight_path = "../checkpoint/xxx"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    with torch.no_grad():
        # predict class
     output = model(img)
     predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)], predict[predict_cla].item())
    plt.show()

if __name__ == '__main__':
    main()

