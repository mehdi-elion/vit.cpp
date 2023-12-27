import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from rich import inspect
import seaborn as sns

# init artifacts folder
out_folder = Path("artifacts/")
out_folder.mkdir(exist_ok=True)

# new size
w = h = 384

# paths to test images
img_dict = {
    "tench": {
        "original": "assets/tench.jpg",
        "resized_bicubic": "build/tench_resized_bicubic.png",
        "resized_bilinear": "build/tench_resized_bilinear.png",
    },
    "armadillo": {
        "original": "assets/armadillo.jpg",
        "resized_bicubic": "build/armadillo_resized_bicubic.png",
        "resized_bilinear": "build/armadillo_resized_bilinear.png",
    },
    "polars": {
        "original": "assets/polars.jpeg",
        "resized_bicubic": "build/polars_resized_bicubic.png",
        "resized_bilinear": "build/polars_resized_bilinear.png",
    },
}




plt.figure(figsize=(35, 25))

# iterate over test images
for i, (key, dico) in enumerate(img_dict.items()):

    # initialize dist dataframes
    df_avg = pd.DataFrame()
    df_max = pd.DataFrame()

    print(f"----- Working with {key} -----")

    # read vit.cpp-resized image (bicubic)
    vit_cpp_resized_img_PIL = Image.open(dico["resized_bicubic"])

    # read original image
    img_path = Path(dico["original"])
    img_PIL = Image.open(dico["original"])
    print(f"Original image read with PIL in '{img_PIL.mode}' mode")

    # resize with pytorch
    transform = T.Resize(
        (w, h), 
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
    )
    torch_resized_img = transform(img_PIL)
    torch_resized_img.save(f"build/{img_path.name}_resized_torch_bicubic.png")

    # resize with opencv
    img_cv2 = cv2.imread(dico["original"], cv2.IMREAD_UNCHANGED)
    cv2_resized_img = cv2.resize(img_cv2, (w, h), cv2.INTER_CUBIC)
    cv2.imwrite(f"build/{img_path.name}_resized_cv2_bicubic.png", cv2_resized_img)

    # compute difference 
    torch_img = np.array(torch_resized_img) / 255.0
    print(torch_img[:3, :3, 0])
    print(torch_img.dtype)
    print("torch_img", torch_img.shape, "\n")

    vit_cpp_img = np.array(vit_cpp_resized_img_PIL) / 255.0
    print(vit_cpp_img[:3, :3, 0])
    print(vit_cpp_img.dtype)
    print("vit_cpp_img", vit_cpp_img.shape, "\n")

    cv2_resized_img = cv2_resized_img[:, :, [2, 1, 0]] / 255.0
    print(cv2_resized_img[:3, :3, 0])
    print(cv2_resized_img.dtype)
    print("cv2_resized_img", cv2_resized_img.shape, "\n")

    
    # fill, plot & save dataframes
    plt.figure(figsize=(35, 35))
    for ii, (name_a, image_a) in enumerate(zip(("torch", "vit.cpp", "cv2"), (torch_img, vit_cpp_img, cv2_resized_img))):
        for jj, (name_b, image_b) in enumerate(zip(("torch", "vit.cpp", "cv2"), (torch_img, vit_cpp_img, cv2_resized_img))):
        
            df_avg.loc[name_a, name_b] = np.sum(np.abs(image_a-image_b)) / (w * h)
            df_max.loc[name_a, name_b] = np.max(np.abs(image_a-image_b))

            plt.subplot(3, 3, 3*ii + jj + 1)
            plt.imshow(
                np.max(np.abs(image_a-image_b), axis=-1),
                cmap="hot",
                vmin=0, vmax=1,
            )
            plt.colorbar()
            plt.title(
                f"{name_a} - {name_b} \n(avg: {df_avg.loc[name_a, name_b]:.4f},  max: {df_max.loc[name_a, name_b]:.4f})",
                fontsize=28,
            )
            if ii==2:
                plt.xlabel(name_b, fontsize=28)
            if jj==0:
                plt.ylabel(name_a, fontsize=28)
    plt.savefig(out_folder / f"{key}_differences.png")
    plt.close()
            

    
    plt.figure()
    sns.heatmap(df_avg, vmin=0, vmax=1, annot=True, fmt=".4f")
    plt.title(f"{key} - Average absolute differences")
    plt.savefig(out_folder / f"{key} - Average absolute differences.png")
    plt.close()

    plt.figure()
    sns.heatmap(df_max, vmin=0, vmax=1, annot=True, fmt=".4f")
    plt.title(f"{key} - Maximum absolute differences")
    plt.savefig(out_folder / f"{key} - Maximum absolute differences.png")
    plt.close()







    # comparative plot (& save)
    plt.subplot(len(img_dict), 4, 4*i + 1)
    plt.imshow(img_PIL)
    plt.title(f"{img_path.name} (orig)")

    plt.subplot(len(img_dict), 4, 4*i + 2)
    plt.imshow(torch_img)
    plt.title(f"torch (bicubic)")

    plt.subplot(len(img_dict), 4, 4*i + 3)
    plt.imshow(vit_cpp_img)
    plt.title(f"vit.cpp (bicubic)")

    plt.subplot(len(img_dict), 4, 4*i + 4)
    map = np.max(np.abs(vit_cpp_img-torch_img), axis=-1)
    total_sum = np.sum(np.abs(vit_cpp_img-torch_img))
    total_max = np.max(np.abs(vit_cpp_img-torch_img))
    plt.imshow(
        # np.max(np.abs(vit_cpp_img-torch_resized_img), axis=-1),
        # (map - map.min()) / (map.max() - map.min()),
        # (map - map.min()) / 255.0,
        map,
        cmap="hot",
        vmin=0, vmax=1,
    )
    plt.colorbar()
    plt.title(f"Difference (avg: {total_sum/(w*h):.4f},  max: {total_max:.4f})")
    print(np.max(np.abs(vit_cpp_img-torch_img), axis=-1))


plt.savefig(out_folder / "Comparison.png")
plt.close()








