import torch
import numpy as np
from AttentionUNET import AttU_Net

model = AttU_Net(in_channels=3, out_channels=2, img_size=(256, 256))


from PIL import Image

image = Image.open('./DSB/images/0e4c2e2780de7ec4312f0efcd86b07c3738d21df30bb4643659962b4da5505a3.jpg')
image = image.resize((256, 256))
image = torch.tensor([np.array(image).transpose((2, 0, 1))])  # convert to tensor and transpose dimensions

with torch.no_grad():
    output = model(image)
    torch.save(output, 'output.pth')

