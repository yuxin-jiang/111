import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Process anomagic and attention module checkpoints")
parser.add_argument("--ckpt_anomagic", type=str, required=True, help="Path to anomagic checkpoint")
parser.add_argument("--ckpt_att", type=str, required=True, help="Path to attention module checkpoint")
args = parser.parse_args()

sd_ip = torch.load(args.ckpt_anomagic, map_location="cpu")

image_proj_sd = {}
ip_sd = {}
unet = {}

for k in sd_ip:
    print(f"Original key name: {k}")
    if k.startswith("unet"):
        unet[k.replace("unet.", "")] = sd_ip[k]
    elif k.startswith("image_proj_model"):
        new_key = k.replace("image_proj_model.", "")
        image_proj_sd[new_key] = sd_ip[k]
        print(f"Processed image_proj key name: {new_key}")
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd_ip[k]

print("\nProcessed image_proj_sd key names:")
for key in image_proj_sd.keys():
    print(f"- {key}")

ckpt_dir = os.path.dirname(args.ckpt_ip)
save_path_ip = os.path.join(ckpt_dir, "anomagic_0.bin")
torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd, "unet": unet}, save_path_ip)
print(f"Model saved to: {save_path_ip}")

sd_att = torch.load(args.ckpt_att, map_location="cpu")

att = {}
for k in sd_att:
    print(k)
    if k.startswith("module"):
        att[k.replace("module.", "")] = sd_att[k]
    else:
        att[k] = sd_att[k]

save_path_att = os.path.join(ckpt_dir, "att.bin")
torch.save({"att": att}, save_path_att)
print(f"Model saved to: {save_path_att}")

print("\nProcessed all key names:")
print("----------------------------------------")
for key in att.keys():

    print(key)
