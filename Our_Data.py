import numpy as np
from PIL import Image,ImageDraw
import random
import matplotlib.pyplot as plt
import math

# Constants
num_groups = 20
matrix_size = 100
shapes = ['circle', 'ellipse', 'triangle']


def create_circle(draw, matrix_size):
    radius = random.randint(matrix_size//5, matrix_size // 3 - 1)
    x = random.randint(radius, matrix_size - radius - 1)
    y = random.randint(radius, matrix_size - radius - 1)
    left_up_point = (x - radius, y - radius)
    right_down_point = (x + radius, y + radius)
    draw.ellipse([left_up_point, right_down_point], outline=1, fill=1)


def create_ellipse(draw, matrix_size):
    x_radius = random.randint(matrix_size//6, matrix_size // 2 - 1)
    y_radius = random.randint(matrix_size//6, matrix_size // 2 - 1)
    while abs(y_radius-x_radius)<=matrix_size//6:
        x_radius = random.randint(matrix_size // 6, matrix_size // 2 - 1)
        y_radius = random.randint(matrix_size // 6, matrix_size // 2 - 1)
    x = random.randint(x_radius, matrix_size - x_radius - 1)#random.randint(x_radius, matrix_size - x_radius - 1)
    y = random.randint(y_radius, matrix_size - y_radius - 1)#random.randint(y_radius, matrix_size - y_radius - 1)
    left_up_point = (x - x_radius, y - y_radius)
    right_down_point = (x + x_radius, y + y_radius)
    draw.ellipse([left_up_point, right_down_point], outline=1, fill=1)

def check_triangle(x1,x2,x3,y1,y2,y3,matrix_size):
    A=np.linalg.norm([x1-x2,y1-y2])
    B=np.linalg.norm([x1-x3,y1-y3])
    C=np.linalg.norm([x2-x3,y2-y3])
    a = math.degrees(math.acos((A * A + B * B - C * C) / (2.0 * A * B)))
    b = math.degrees(math.acos((C * C + B * B - A * A) / (2.0 * C * B)))
    c = 180-a-b

    if a>120 or b>120 or c>120 or A<matrix_size/5 or B<matrix_size/5 or C<matrix_size/5:
        return True
    else:
        return False




def create_triangle(draw, matrix_size):
    while True:
        x1, y1 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
        x2, y2 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
        x3, y3 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
        check_1 = check_triangle(x1, x2, x3, y1, y2, y3,matrix_size)
        while check_1==True:
            x1, y1 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
            x2, y2 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
            x3, y3 = random.randint(0, matrix_size - 1), random.randint(0, matrix_size - 1)
            check_1 = check_triangle(x1, x2, x3, y1, y2, y3,matrix_size)

        if (0 <= x1 < matrix_size and 0 <= y1 < matrix_size and
                0 <= x2 < matrix_size and 0 <= y2 < matrix_size and
                0 <= x3 < matrix_size and 0 <= y3 < matrix_size):
            break

    draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline=1, fill=1)


def create_shape(shape, matrix_size):
    img = Image.new('L', (matrix_size, matrix_size), 0)
    draw = ImageDraw.Draw(img)
    if shape == 'circle':
        create_circle(draw, matrix_size)
    elif shape == 'ellipse':
        create_ellipse(draw, matrix_size)
    elif shape == 'triangle':
        create_triangle(draw, matrix_size)
    return img  # Return the PIL Image object


def create_groups(num_groups, shapes, matrix_size):
    groups = []
    for _ in range(num_groups):
        group = []
        for shape in shapes:
            img = create_shape(shape, matrix_size)
            group.append(np.array(img))  # Convert PIL Image to numpy array
        groups.append(group)
    return groups


# Create the groups
groups = create_groups(num_groups, shapes, matrix_size)

# Convert images to binary representation
binary_groups = []
for group in groups:
    binary_group = []
    for img in group:
        # Convert image to binary representation
        binary_img = np.where(img == 0, 0, 1)  # 0 represents white, 1 represents black
        binary_group.append(binary_img)
    binary_groups.append(binary_group)


# Display all groups
def display_groups(groups):
    fig, axs = plt.subplots(num_groups, len(shapes), figsize=(12, num_groups * 2))
    for i, group in enumerate(groups):
        for j, (img, shape) in enumerate(zip(group, shapes)):
            ax = axs[i, j]
            ax.imshow(img, cmap='gray')
            ax.set_title(shape)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def display_binary_groups(groups):
    fig, axs = plt.subplots(num_groups, len(shapes), figsize=(12, num_groups * 2))
    for i, group in enumerate(groups):
        for j, (img, shape) in enumerate(zip(group, shapes)):
            ax = axs[i, j]
            ax.imshow(img, cmap='binary')  # Use binary colormap
            ax.set_title(shape)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def save_groups(groups, file_prefix):
    for i, group in enumerate(groups):
        for j, img in enumerate(group):
            if j == 0:
                filename = f"{file_prefix}_group_{i}_shape_of_Circle.png"
            elif j == 1:
                filename = f"{file_prefix}_group_{i}_shape_of_Ellipse.png"
            elif j == 2:
                filename = f"{file_prefix}_group_{i}_shape_of_triangle.png"

            plt.imsave(filename, img, cmap='binary')


import os


def Photo_to_Binary_list(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    # Open the image
    img = Image.open(image_path).convert("L")  # Ensure the image is in grayscale
    pixels = img.load()
    width, height = img.size
    # Convert to binary
    binary_photo = [
        [
            int(pixels[x, y] < 125)
            for y in range(height)
        ]
        for x in range(width)

    ]

    for row in binary_photo:
        print(row)

display_groups(groups)
save_groups(groups, "Set of groups")