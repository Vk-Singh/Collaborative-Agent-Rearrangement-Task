import numpy as np
from PIL import Image

n_image = np.load("leader_scene.npy")

img = Image.fromarray(n_image)
img.show()
