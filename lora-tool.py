#!/usr/bin/env python3

import argparse
import os
import shutil
from rich.console import Console

PATH_MY_KOHYA = "/home/fortyseven/opt/ai/kohya_ss"

console = Console()

args = None

configs = {
    "sd15": {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "bucket_reso_steps": 64,
        "caption_extension": ".txt",
        "learning_rate": "0.0001",
        "lr_scheduler_num_cycles": "3",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 220,
        "max_data_loader_n_workers": 0,
        # "max_train_steps" : "22000",
        "mixed_precision": "bf16",
        # "network_dim" : 256,
        "network_dim": 16,
        "networks_module": "lycoris.kohya",
        "noise_offset": 0.0,
        "optimizer_args": "",
        "optimizer_type": "AdamW8bit",
        "res": "512,512",
        "save_every_n_epochs": "10",
        "save_precision": "bf16",
        "script": "./train_network.py",
        "text_encoder_lr": "5e-05",
        "train_batch_size": "4",
        "unet_lr": 0.0001,
    },
    "sdxl": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "bucket_reso_steps": 64,
        "caption_extension": ".txt",
        "learning_rate": "0.0004",
        "lr_scheduler_num_cycles": "10",
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_data_loader_n_workers": 0,
        # "max_train_steps" : "22000",
        "mixed_precision": "bf16",
        "network_dim": 64,  # ends up being just over ~450 megs, 256 is almost 2 gigs with not enough benefit
        "networks_module": "networks.lora",
        "noise_offset": 0.0,
        "optimizer_args": "--optimizer_args scale_parameter=False relative_step=False warmup_init=False",
        "optimizer_type": "Adafactor",
        "res": "1024,1024",
        "save_every_n_epochs": "10",
        "save_precision": "bf16",
        "script": "./sdxl_train_network.py",
        "text_encoder_lr": "0.0004",
        "train_batch_size": "1",
        "unet_lr": 0.0004,
    },
}


def is_project_path_valid(path: str) -> bool:
    """
    Checks if the path is valid for a project, meaning it should be
    a directory and contain at least one image.
    """

    if not os.path.exists(path):
        console.log("[red]Path does not exist[/red]")
        exit(1)

    if not os.path.isdir(path):
        console.log("[red]Path is not a directory[/red]")
        exit(1)

    if not any(
        [
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
            for filename in os.listdir(path)
        ]
    ):
        console.log(f"[red]Path `{path}` contains no images[/red]")
        return False
    return True


def rename_batch(source_path: str, keyword: str) -> None:
    from rich.table import Table

    table = Table(title=f"Renaming Images in {source_path}")
    table.add_column("Old Name")
    table.add_column("New Name")

    # get all images
    images = [
        filename
        for filename in os.listdir(source_path)
        if filename.endswith(".jpg")
        or filename.endswith(".jpeg")
        or filename.endswith(".png")
    ]
    images.sort()

    # rename images
    for index, image in enumerate(images):
        extension = os.path.splitext(image)[1]
        new_name = f"{index+1:04d}-{keyword}{extension}"

        # rename file
        orig_path = os.path.join(source_path, image)
        new_path = os.path.join(source_path, new_name)
        os.rename(orig_path, new_path)

        table.add_row(image, new_name)

    console.print(table)


def generate_caption_files(source_path: str) -> None:
    # iterate through all images and create a caption file for each
    # the caption file is a .txt file of the same filename as the image
    # inside the file is the caption text comprised of the filename without
    # the extension and without the 0001- prefix

    images = [
        filename
        for filename in os.listdir(source_path)
        if filename.endswith(".jpg")
        or filename.endswith(".jpeg")
        or filename.endswith(".png")
    ]
    images.sort()

    for image in images:
        caption_filename = f"{os.path.splitext(image)[0]}.txt"
        caption_path = os.path.join(source_path, caption_filename)
        index_split_index = os.path.splitext(image)[0].index("-") + 1
        caption_text = os.path.splitext(image)[0][index_split_index:]

        # console.log(f"[green]Regenerating caption file for {image}...[/green]")

        with open(caption_path, "w") as f:
            f.write(caption_text)

    console.log(f"[green]Caption files regenerated.[/green]")


def _build_launch_str():
    if not args:
        raise ValueError("args is not set")

    sd15 = args.sd15

    config = configs["sd15"] if sd15 else configs["sdxl"]

    console.log(f"[green]Using {'sd15' if sd15 else 'sdxl'  } config: {config}[/green]")

    return f"""
    accelerate launch
        --num_cpu_threads_per_process=12
        "{config['script'] }"
        --enable_bucket
        --min_bucket_reso=256
        --max_bucket_reso=2048
        --pretrained_model_name_or_path="{config['base_model']}"
        --train_data_dir="{args.path_lora_images}"
        --output_dir="{args.path_lora_model}"
        --logging_dir="{args.path_lora_logs}"
        --resolution="{config['res']}"
        --network_alpha="1"
        --network_module="{config['networks_module']}"
        --max_train_epochs="30"
        --save_model_as=safetensors
        --network_args "conv_dim=1" "conv_alpha=1" "algo=lora"
        --text_encoder_lr={config['text_encoder_lr']}
        --unet_lr={config['unet_lr']}
        --network_dim={config['network_dim']}
        --output_name="last"
        --lr_scheduler_num_cycles="{config['lr_scheduler_num_cycles']}"
        --lr_scheduler={config['lr_scheduler']}
        --lr_warmup_steps="{config['lr_warmup_steps']}"
        --no_half_vae
        --learning_rate={config['learning_rate']}
        --train_batch_size={config['train_batch_size']}
        --save_every_n_epochs={config['save_every_n_epochs']}
        --mixed_precision="{config['mixed_precision']}"
        --save_precision="{config['save_precision']}"
        --caption_extension="{config['caption_extension']}"
        --cache_latents
        --optimizer_type="{config['optimizer_type']}"
        --max_data_loader_n_workers="{config['max_data_loader_n_workers']}"
        --bucket_reso_steps={config['bucket_reso_steps']}
        --xformers
        --bucket_no_upscale
        --noise_offset={config['noise_offset']}
        {config['optimizer_args']}
    """.replace(
        "\n", " "
    )


def run_lora_process(source_path: str) -> None:
    global args

    if not args.keyword:  # type: ignore
        console.log("[red]Keyword is required[/red]")
        exit(1)

    generate_caption_files(source_path)

    args.path_lora = os.path.join(source_path, "lora")  # type: ignore
    os.makedirs(args.path_lora, exist_ok=True)  # type: ignore

    args.path_lora_model = os.path.join(source_path, "lora/model")  # type: ignore
    os.makedirs(args.path_lora_model, exist_ok=True)  # type: ignore

    args.path_lora_logs = os.path.join(source_path, "lora/logs")  # type: ignore
    os.makedirs(args.path_lora_logs, exist_ok=True)  # type: ignore

    args.path_lora_images = os.path.join(source_path, "lora/images")  # type: ignore
    if os.path.exists(args.path_lora_images):  # type: ignore
        shutil.rmtree(args.path_lora_images)  # type: ignore

    os.makedirs(args.path_lora_images, exist_ok=True)  # type: ignore

    ####

    args.path_lora_image_label = os.path.join(source_path, f"lora/images/{args.repeats}_{args.keyword}")  # type: ignore
    os.makedirs(args.path_lora_image_label, exist_ok=True)  # type: ignore

    for filename in os.listdir(source_path):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
            or filename.endswith(".txt")
        ):
            shutil.copy(
                os.path.join(source_path, filename),
                args.path_lora_image_label,  # type: ignore
            )

    try:
        # push
        start_dir = os.getcwd()
        os.chdir(PATH_MY_KOHYA)

        console.log("Executing Kohya_ss process...")
        os.system(f". ./venv/bin/activate && {_build_launch_str()} && deactivate")
        # print(f". ./venv/bin/activate && {_build_launch_str()} && deactivate")
    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")
    finally:
        # pop
        console.log("Returning home")
        os.chdir(start_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Tool (for use with `kohya_ss`)")

    # default argument is a path that defaults to the current directory
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project path (defaults to current directory)",
    )

    # required keyword argument
    parser.add_argument(
        "-k",
        "--keyword",
        required=True,
        help="Keyword for the LoRA",
    )

    # optional sd15 bool
    parser.add_argument(
        "--sd15",
        action="store_true",
        help="Use SD15 model (defaults to SDXL))",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    parser.add_argument(
        "--rename-images",
        action="store_true",
        help="Rename images in the project directory with a prefix (e.g. 0001-keyword.jpg)",
    )

    # optional integer 'repeats' argument
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=100,
        help="Number of repeats for the LoRA",
    )

    # parser.add_argument(
    #     "-g",
    #     "--regen-caption-files",
    #     action="store_true",
    #     help="Regenerate caption files for all images in the project directory",
    # )

    args = parser.parse_args()
    args.path = os.path.abspath(args.path)

    # print(args)

    if not is_project_path_valid(args.path):
        exit(1)

    console.log(f"[green]Path `{args.path}` is valid.[/green]")

    # we have some one-off operations that won't kick off further processing
    if args.rename_images:
        rename_batch(args.path, args.keyword)
        exit(0)

    # run lora building process

    run_lora_process(args.path)
