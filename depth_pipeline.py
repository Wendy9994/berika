import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# PyTorch backend
import torch
from torch import autocast

# Transformers containing tools
from transformers import CLIPTextModel, CLIPTokenizer, DPTForDepthEstimation, DPTFeatureExtractor

# Diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler

# Customized Diffusion Pipeline Class
class DiffusionPipeline:
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler):
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_text_embeds(self, text):
        text_input = self.tokenizer(text,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeds

    def get_prompt_embeds(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        cond_embeds = self.get_text_embeds(prompt)
        uncond_embeds = self.get_text_embeds([''] * len(prompt))
        return torch.cat([uncond_embeds, cond_embeds])

    def decode_img_latents(self, img_latents):
        img_latents = 1 / self.vae.config.scaling_factor * img_latents
        with torch.no_grad():
            img = self.vae.decode(img_latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        return img

    def transform_img(self, img):
        img = (img * 255).round().astype('uint8')
        return [Image.fromarray(i) for i in img]

    def encode_img_latents(self, img, latent_timestep):
        img = np.stack([np.array(i) for i in img], axis=0)
        img = 2 * ((img / 255.0) - 0.5)
        img = torch.from_numpy(img).float().permute(0, 3, 1, 2).to(self.device)
        img_latents_dist = self.vae.encode(img)
        img_latents = img_latents_dist.latent_dist.sample()
        img_latents = self.vae.config.scaling_factor * img_latents
        noise = torch.randn(img_latents.shape).to(self.device)
        return self.scheduler.add_noise(img_latents, noise, latent_timestep)

class Depth2ImgPipeline(DiffusionPipeline):
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler, depth_feature_extractor, depth_estimator):
        super().__init__(vae, tokenizer, text_encoder, unet, scheduler)
        self.depth_feature_extractor = depth_feature_extractor
        self.depth_estimator = depth_estimator

    def get_depth_mask(self, img):
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((img * 255).astype('uint8'))

        if not isinstance(img, list):
            img = [img]

        width, height = img[0].size
        pixel_values = self.depth_feature_extractor(img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with autocast(self.device):
            depth_mask = self.depth_estimator(pixel_values).predicted_depth

        depth_mask = torch.nn.functional.interpolate(depth_mask.unsqueeze(1),
                                                     size=(height // 8, width // 8),
                                                     mode='bicubic', align_corners=False)
        depth_min = torch.amin(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_mask = 2.0 * (depth_mask - depth_min) / (depth_max - depth_min) - 1.0
        return torch.cat([depth_mask] * 2)

    def denoise_latents(self, img, prompt_embeds, depth_mask, strength, num_inference_steps=20, guidance_scale=7.5, height=512, width=512):
        strength = max(min(strength, 1), 0)
        self.scheduler.set_timesteps(num_inference_steps)
        init_timestep = int(num_inference_steps * strength)
        timesteps = self.scheduler.timesteps[init_timestep:]
        latents = self.encode_img_latents(img, timesteps[:1].repeat(1))

        with autocast(self.device):
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    def __call__(self, prompt, img, strength=0.8, num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
        prompt_embeds = self.get_prompt_embeds(prompt)
        depth_mask = self.get_depth_mask(img)
        latents = self.denoise_latents(img, prompt_embeds, depth_mask, strength, num_inference_steps, guidance_scale, height, width)
        img = self.decode_img_latents(latents)
        return self.transform_img(img)
