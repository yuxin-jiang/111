import os
from peft import LoraConfig, LoraModel
import torch
from typing import List, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
import torch.nn as nn
import math
from .utils import is_torch2_available, get_generator
import numpy as np

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


def load_lora_model(unet, device, diffusion_model_learning_rate):
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=diffusion_model_learning_rate,
    )
    return unet, lora_layers


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        b = embeds.shape[0]
        clip_extra_context_tokens = self.proj(embeds).reshape(
            b, -1, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class SelfAttention(nn.Module):
    def __init__(self, in_channels, device):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.proj_out = nn.Linear(1280, 1024).to(device)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        batch_size, channels, h = x.size()
        height = int(math.sqrt(h))
        width = height
        x = x.view(batch_size, channels, width, height)
        batch_size, channels, height, width = x.size()
        
        # 计算 query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # 计算注意力分数
        attention_scores = torch.bmm(q, k)

        if mask is not None:
            # 将 mask 的尺寸调整为和 x 一致
            mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest')
            mask = mask.view(batch_size, 1, height * width)
            large_constant = 1e6
            attention_scores = attention_scores - (1 - mask) * large_constant

        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)

        # 应用注意力权重
        out = torch.bmm(v, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # 加权求和
        out = self.gamma.to(x.device) * out + x
        out = out.view(batch_size, channels, height * width)
        out = out.permute(0, 2, 1)
        out = self.proj_out(out)

        return out


class Anomagic:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, ip_ckpt_1, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.ip_ckpt_1 = ip_ckpt_1
        self.num_tokens = num_tokens
        self.attention_module = SelfAttention(1280, device)
        self.pipe = sd_pipe.to(self.device)
        self.set_anomagic()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_anomagic()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

        return image_proj_model

    def set_anomagic(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        unet, lora_layers = load_lora_model(unet, self.device, 4e-4)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_anomagic(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter_model."):
                        state_dict["ip_adapter"][key.replace("ip_adapter_model.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        self.pipe.unet.load_state_dict(state_dict["unet"])
        state_dict_1 = torch.load(self.ip_ckpt_1, map_location="cpu")
        self.attention_module.load_state_dict(state_dict_1["att"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, mask_image_0=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            outputs = self.image_encoder(clip_image.to(self.device, dtype=torch.float16))
            clip_image_embeds = outputs.image_embeds
            last_feature_layer_output = outputs.last_hidden_state
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        mask_image_0 = mask_image_0.resize((64, 64))
        mask_image_0 = mask_image_0.convert('L')
        mask_image_0 = torch.tensor(np.array(mask_image_0), dtype=torch.float32)
        mask_image_0 = (mask_image_0 > 0.5).float().to(self.device)

        global_image_embeds = image_embeds.mean(dim=1)

        image_prompt_embeds = self.image_proj_model(global_image_embeds.half())
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(global_image_embeds).half())

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def encode_long_text(self,
            input_ids: torch.Tensor,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            max_length: int = 77,
            device: str = "cuda"
    ) -> torch.Tensor:
        """
        分段编码长文本并合并
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size = input_ids.size(0)
        hidden_dim = text_encoder.config.hidden_size
        combined_embeddings = torch.zeros(batch_size, hidden_dim, device=device)

        for batch_idx in range(batch_size):
            current_input_ids = input_ids[batch_idx]
            chunks = [
                current_input_ids[i:i + max_length]
                for i in range(0, len(current_input_ids), max_length)
            ]

            embeddings = []
            for chunk in chunks:
                chunk_len = len(chunk)
                padding_len = max_length - chunk_len

                chunk_input = {
                    "input_ids": torch.cat([
                        chunk.unsqueeze(0).to(device),
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)
                    ], dim=1),
                    "attention_mask": torch.cat([
                        torch.ones(1, chunk_len, dtype=torch.long, device=device),
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)
                    ], dim=1)
                }

                with torch.no_grad():
                    chunk_emb = text_encoder(**chunk_input).last_hidden_state
                    embeddings.append(chunk_emb[:, :chunk_len, :].mean(dim=1))

            if embeddings:
                combined_embeddings[batch_idx] = torch.mean(torch.cat(embeddings, dim=0), dim=0)
            else:
                combined_embeddings[batch_idx] = torch.zeros(hidden_dim, device=device)

        return combined_embeddings.unsqueeze(1)

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            mask_image_0=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, mask_image_0=mask_image_0,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_list = []
            for p in prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)

                prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )

                prompt_embeds_list.append(prompt_embed)

            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

            negative_prompt_embeds_list = []
            for p in negative_prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)

                negative_prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )

                negative_prompt_embeds_list.append(negative_prompt_embed)

            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)

            # 合并图像嵌入与文本嵌入
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class AnomagicXL(Anomagic):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class AnomagicPlus(Anomagic):
    """Anomagic with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class AnomagicFull(AnomagicPlus):
    """Anomagic with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class AnomagicPlusXL(Anomagic):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
