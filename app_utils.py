from torch.nn.functional import interpolate
from models import VAE
import torch

from torchvision.transforms import Resize, ToPILImage, Compose
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def canvas_to_tensor(canvas):
	img = canvas.image_data
	img = img[:, :, :-1]
	img = img.mean(axis=2)
	img /= 255
	# img = img*2 - 1
	img = torch.FloatTensor(img)
	tens = img.unsqueeze(0).unsqueeze(0)
	tens = interpolate(tens, (28, 28))
	# plt.imshow(tens.detach().squeeze())
	# plt.savefig("./f.png")
	return tens

MODEL_DIR = "./check_points/"
device = "cpu"

def load_model(filename):
	model_type = "conv_vae"
	model = VAE().to("cpu")
	model.load_state_dict(torch.load(MODEL_DIR+filename))
	return model

def resize_img(img, w, h):
	return img.resize((w, h))


def tensor_to_img(tens):
	if tens.ndim == 4:
		tens = tens.squeeze()
	transform = Compose([
		ToPILImage()

		])
	img = transform(tens)
	return img

def interpolation(model, inp1, inp2):
	model = model.eval()
	output = model.interpolate(inp1, inp2)
	output = (output+1)/2
	grid = make_grid(output, nrow=10)
	img = tensor_to_img(grid)
	return resize_img(img, 1500, 300)
