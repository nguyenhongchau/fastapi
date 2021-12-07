from PIL import Image
import requests
from io import BytesIO
import torch
from torch.nn.functional import softmax
from nets.net import MNISTNet, transform


serving_model = MNISTNet()

def round_up(f):
    import math
    return math.ceil(f * 1000) / 1000

def do_inference(url):
    serving_model.eval()
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except :
        return None, None # no image found
    
    img = transform(img)
    img = img[None, :]
    with torch.no_grad():
        sm = softmax(serving_model(img))
        pred = sm.argmax(dim=1, keepdim=True).item()
        prob = sm[0,pred].item()
        return pred, round_up(prob)
