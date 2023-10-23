# LoRA Tool

This is just a simple Python script that works WITH and *requires* the [Kohya_SS project](https://github.com/bmaltais/kohya_ss) to be installed and working.

This is a tool meant for my own use, but it might end up being useful for others, so why not put it out there?

(This means I'm not offering active support, since this isn't a full "project".)


## What the hell is it?

I've been trying to minimize the effort taken to create a high quality LoRA model.

This basically means:
- Collecting quality images for your LoRA to train on.
- Creating the destination image, log, and output directories.
- Copying all the images to a properly formatted directory name.
- Optionally adding caption files.
- Running the Kohya scripts with all the appropriate paths.

Initially, I used the Kohya UI itself, and it was good.

But quickly I found the gradio interface a bit irritating to use and useful tools were scattered around the app. It worked out for a bit, but it wasn't as automatic as I'd have preferred.

Then I realized Kohya let you see the command it was generating to create the LoRA, so I took that and wrote a shell script to automate some of the tasks.

That went well for a period of time, but there was still a lot of room for improvement. So I took the script and recreated everything in a nice Python app and added better support for caption files.


## Using it

First, you'll need to update `PATH_MY_KOHYA` to match where it exists on your system.

Then, you just take a directory full of curated png/jpg files.

You run `lora-tool.py --rename-images -k my_keyword`.

This will rename all of the images into a sequential pattern like this:
```
0001-my_keyword.png
0002-my_keyword.jpeg
0003-my_keyword.png
0004-my_keyword.jpg
0005-my_keyword.png
```

Next, I rename the files as the _caption_ I want. (Adding the keyword into the filename to start you off.)

By making the filename the caption text, it makes it easier to simply use a basic file manager to look at the image and rename it, instead of having to edit a separate `.txt` file.

When you begin generating the LoRA, part of that process is to generate all the .txt files along side the images (with the caption text inside them).

Here's a real life example of what gets generated once LoRA processing begins:
```
0001-tng-worf\ looking\ up,\ concerned.png
0001-tng-worf\ looking\ up,\ concerned.txt
0002-tng-worf\ looking\ to\ the\ side.png
0002-tng-worf\ looking\ to\ the\ side.txt
0003-tng-worf\ looking\ ahead,\ pleased.png
0003-tng-worf\ looking\ ahead,\ pleased.txt
0004-tng-worf\ looking\ straight\ ahead.png
0004-tng-worf\ looking\ straight\ ahead.txt
0005-tng-worf\ wearing\ red,\ looking\ off\ to\ the\ side,\ suspicious.png
0005-tng-worf\ wearing\ red,\ looking\ off\ to\ the\ side,\ suspicious.txt
```

Using captions is entirely optional, but strongly recommended to increase the flexibility of generated imagery. (If it knows what Worf looks like smiling, for instance, it has an example to draw from instead of just extrapolating from known concepts. At least, that's how I understand it.)

Once the images are prepped, just run `lora-tool.py -k my_keyword` in the directory with the images, and it will create all the infrastructure for you in the `/lora` directory, with generated models in `/lora/model/*.safetensors`.

# Notes
- This script currently has SDXL hard coded in a bool as `True`, since SDXL generation has been my focus since making this. Now that I understand the generation parameters a bit better, I may revisit SD15 LoRAs again later.
- **This script really isn't intended for the mainstream, so some obvious improvements (like adding a switch for the aforementioned SDXL mode) are missing.**
  - Those interested in it, or whatever reason, would be better served by just forking this and customizing it for their own uses, since I have no interest in actively maintaining this. Especially since it's basically just a glorified shell script around someone else's project.