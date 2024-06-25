import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_fully_connected_graph_from_image(image):
    rows, cols = image.shape
    G = nx.Graph()
    
    # Add all pixels as nodes
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), intensity=image[r, c])
    
    # Connect each pixel to all other pixels
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(rows):
                for c2 in range(cols):
                    if (r1, c1) != (r2, c2):
                        weight = abs(int(image[r1, c1]) - int(image[r2, c2]))
                        G.add_edge((r1, c1), (r2, c2), weight=weight)
    
    return G

def find_borders(G, threshold=30):
    edges = []
    for (u, v, d) in G.edges(data=True):
        if d['weight'] > threshold:
            edges.append((u, v))
    return edges

def draw_borders(image, edges):
    for (u, v) in edges:
        image[u] = 255
        image[v] = 255
    return image

image_path = 'imagens_teste/folha-low.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



G = create_fully_connected_graph_from_image(image)

# Find the borders based on edge weights
edges = find_borders(G)

# Draw the borders on a blank image
border_image = np.zeros_like(image)
border_image = draw_borders(border_image, edges)

# Display the original and border images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Borders')
plt.imshow(border_image, cmap='gray')
plt.savefig("papito.jpg")
plt.show()