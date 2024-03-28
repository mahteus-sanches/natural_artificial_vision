import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters

image = io.imread('/home/vacilo/github/natural_artificial_vision/southern_ring_nebula.png')
if image.ndim == 2:
    image_matrix = np.array(image)
elif image.ndim == 3:
    image_matrix = np.array(image)

print('Shape of the matrix:', image_matrix.shape)
print('Image matrix:')
print(image_matrix)

plt.imshow(image)
plt.axis('off')
plt.show()

