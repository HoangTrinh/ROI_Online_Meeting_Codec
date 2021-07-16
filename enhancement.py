import torch
import model
from test import opt
from torchvision import transforms
import imageio
import model
from utils import quantize
from torch.autograd import Variable


#loader = transforms.Compose([transforms.ToTensor()])
device = torch.device('cpu' if opt.cpu else 'cuda:'+ str(opt.cudaDevice))

def loadImage(image):
    """load image, returns cuda tensor"""
    #image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    #image = image.unsqueeze(0)
    return image#.to(device)

def enhanceBG(name, cor, bgModel):
    BU = imageio.imread(name)
    xmin,xmax,ymin,ymax = cor
    image = BU[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1,:]
    image = loadImage(image)
    print("Starting the BU Model")
    image = bgModel(image)
    print("End the BU Model")
    image = quantize(image, 1)
    image = image.mul(255)
    tensor_cpu = image.byte().permute(1, 2, 0).cpu().numpy()
    BU[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1,:] = tensor_cpu
    print("End the CPU tasks")
    return BU

def enhanceOJ(opt, OJ,cor, preBU, ojModel, nNeibs = 5):
    #preBU = imageio.imread(preBU)
    #OJ = imageio.imread(name)

    xmin,xmax,ymin,ymax = cor

    BU = preBU

    BU[int(ymin):int(ymax)+1,int(xmin):int(xmax)+1,:] = OJ

    if (xmin - nNeibs) < 0: xmin = 0
    else: xmin = xmin - nNeibs

    if (ymin - nNeibs) < 0: ymin = 0
    else: ymin = ymin - nNeibs

    if (xmax + nNeibs) >= opt.width: xmax =  opt.width -1
    else: xmax = xmax + nNeibs

    if (ymax + nNeibs) >= opt.height: ymax = opt.height -1
    else: ymax = ymax + nNeibs

    image = BU[ymin:ymax+1,xmin:xmax+1,:]

    image = loadImage(image)

    image = ojModel(image)
    print("Starting the RU Model")
    image = quantize(image, 1)
    print("End the RU Model")
    image = image.mul(255)
    tensor_cpu = image.byte().permute(1, 2, 0).cpu().numpy()
    BU[ymin:ymax+1,xmin:xmax+1,:] = tensor_cpu
    print("End the CPU tasks")
    return BU
