# read slm phase and generate masked images on target planes

from prop_model import CNNpropCNN_default
from PIL import Image
from torchvision import transforms

input_phase_path = 'slm_phase/0043_None.png'

input_phase = Image.open(input_phase_path)

tf = transforms.Compose([
    # transforms.ToTensor()
    transforms.PILToTensor()
])

input_tensor = tf(input_phase)

CNNpropCNN = CNNpropCNN_default()

output_field = CNNpropCNN(input_tensor)

output_amp = output_field.abs()

images = []
for i in range(8):
    images.append(output_amp.squeeze()[i,:,:])

pass