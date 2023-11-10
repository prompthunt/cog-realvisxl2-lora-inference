import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

import json
import requests
from io import BytesIO
import tarfile
import torch
from PIL import Image
import shutil
import math


from dataset_and_utils import TokenEmbeddingsHandler

from image_processing import (
    face_mask_google_mediapipe,
    crop_faces_to_square,
    paste_inpaint_into_original_image,
)

from gfpgan import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import cv2

MODEL_NAME = "SG161222/RealVisXL_V2.0"
MODEL_CACHE = "model-cache"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_output(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def upscale_image_pil(self, img: Image.Image) -> Image.Image:
        weight = 0.5
        try:
            # Convert PIL Image to numpy array if necessary
            img = np.array(img)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            # Enhance the image using GFPGAN
            _, _, output = self.face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )

            # Convert numpy array back to PIL Image
            output = Image.fromarray(output)

            return output

        except Exception as error:
            print("An exception occurred:", error)
            raise

    def load_trained_weights(self, weights_url, pipe):
        # Get the TAR archive content
        weights_tar_data = requests.get(weights_url).content
        with tarfile.open(fileobj=BytesIO(weights_tar_data), mode="r") as tar_ref:
            tar_ref.extractall("trained-model")

        local_weights_cache = "./trained-model"
        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. Assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            sd = pipe.unet.state_dict()
            sd.update(new_unet_params)
            pipe.unet.load_state_dict(sd)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet = pipe.unet
            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                module = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=name_rank_map[name],
                )
                unet_lora_attn_procs[name] = module.to("cuda")

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False

        if not os.path.exists("gfpgan/weights/realesr-general-x4v3.pth"):
            os.system(
                "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights"
            )
        if not os.path.exists("gfpgan/weights/GFPGANv1.4.pth"):
            os.system(
                "wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights"
            )

        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        model_path = "gfpgan/weights/realesr-general-x4v3.pth"
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

        # Use GFPGAN for face enhancement
        self.face_enhancer = GFPGANer(
            model_path="gfpgan/weights/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )
        self.current_version = "v1.4"

        print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        # Copy the image to a temporary location
        shutil.copyfile(path, "/tmp/image.png")
        # Open the copied image
        img = Image.open("/tmp/image.png")
        # Calculate the new dimensions while maintaining aspect ratio
        width, height = img.size
        new_width = math.ceil(width / 64) * 64
        new_height = math.ceil(height / 64) * 64
        # Resize the image if needed
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Convert the image to RGB mode
        img = img.convert("RGB")
        return img

    @torch.inference_mode()
    def predict(
        self,
        lora_url: str = Input(
            description="Load Lora model",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="A photo of TOK",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        pose_image: Path = Input(
            description="Input pose image for controlnet mode",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPMSolverMultistep",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        mask_blur_amount: float = Input(
            description="Amount of blur to apply to the mask.", default=8.0
        ),
        face_padding: float = Input(
            description="Amount of padding (as percentage) to add to the face bounding box.",
            default=0.5,
        ),
        face_resize_to: int = Input(
            description="Resize the face bounding box to this size (in pixels).",
            default=768,
        ),
        inpaint_prompt: str = Input(
            description="Input inpaint prompt", default="A photo of TOK"
        ),
        inpaint_negative_prompt: str = Input(
            description="Input inpaint negative prompt",
            default="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)",
        ),
        inpaint_strength: float = Input(
            description="Prompt strength when using inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.35,
        ),
        inpaint_num_inference_steps: int = Input(
            description="Number of denoising steps for inpainting",
            ge=1,
            le=500,
            default=25,
        ),
        inpaint_guidance_scale: float = Input(
            description="Scale for classifier-free guidance for inpainting",
            ge=1,
            le=50,
            default=3.0,
        ),
    ) -> List[Path]:
        # Check if there is a lora_url
        if lora_url == None:
            raise Exception(f"Missing Lora_url parameter")

        lora = True
        if lora == True:
            self.is_lora = True
            print("LORA")
            print("Loading ssd txt2img pipeline...")
            self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            print("Loading controlnet model")
            self.controlnet = ControlNetModel.from_pretrained(
                "thibaud/controlnet-openpose-sdxl-1.0",
                torch_dtype=torch.float16,
                cache_dir="/src/controlnet-cache",
            )
            self.controlnet.to("cuda")

            print("Loading controlnet pipeline")
            self.controlnet_pipe_txt2img = StableDiffusionXLControlNetPipeline(
                controlnet=self.controlnet,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
                vae=self.txt2img_pipe.vae,
            )

            print("Loading controlnet inpaint pipeline")
            self.controlnet_pipe_inpaint = StableDiffusionXLControlNetInpaintPipeline(
                controlnet=self.controlnet,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
                vae=self.txt2img_pipe.vae,
            )

            print("Loading SDXL inpaint pipeline...")
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )

            # TODO: LOAD ALL PIPELINE LORA WEIGHTS
            print("Loading ssd lora weights...")
            self.load_trained_weights(lora_url, self.txt2img_pipe)
            self.load_trained_weights(lora_url, self.controlnet_pipe_txt2img)
            self.load_trained_weights(lora_url, self.controlnet_pipe_inpaint)
            self.load_trained_weights(lora_url, self.inpaint_pipe)
            self.txt2img_pipe.to("cuda")
            self.controlnet_pipe_txt2img.to("cuda")
            self.controlnet_pipe_inpaint.to("cuda")
            self.inpaint_pipe.to("cuda")
            self.is_lora = True

            # print("Loading SDXL img2img pipeline...")
            # self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            #     vae=self.txt2img_pipe.vae,
            #     text_encoder=self.txt2img_pipe.text_encoder,
            #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
            #     tokenizer=self.txt2img_pipe.tokenizer,
            #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
            #     unet=self.txt2img_pipe.unet,
            #     scheduler=self.txt2img_pipe.scheduler,
            # )
            # self.img2img_pipe.to("cuda")

            # print("Loading SDXL refiner pipeline...")

            # print("Loading refiner pipeline...")
            # self.refiner = DiffusionPipeline.from_pretrained(
            #     "refiner-cache",
            #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
            #     vae=self.txt2img_pipe.vae,
            #     torch_dtype=torch.float16,
            #     use_safetensors=True,
            #     variant="fp16",
            # )
            # self.refiner.to("cuda")

        else:
            print("Loading sdxl txt2img pipeline...")
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.is_lora = False

            self.txt2img_pipe.to("cuda")

            # print("Loading SDXL img2img pipeline...")
            # self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            #     vae=self.txt2img_pipe.vae,
            #     text_encoder=self.txt2img_pipe.text_encoder,
            #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
            #     tokenizer=self.txt2img_pipe.tokenizer,
            #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
            #     unet=self.txt2img_pipe.unet,
            #     scheduler=self.txt2img_pipe.scheduler,
            # )
            # self.img2img_pipe.to("cuda")

            # print("Loading SDXL inpaint pipeline...")
            # self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            #     vae=self.txt2img_pipe.vae,
            #     text_encoder=self.txt2img_pipe.text_encoder,
            #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
            #     tokenizer=self.txt2img_pipe.tokenizer,
            #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
            #     unet=self.txt2img_pipe.unet,
            #     scheduler=self.txt2img_pipe.scheduler,
            # )
            # self.inpaint_pipe.to("cuda")
            # print("Loading refiner pipeline...")
            # self.refiner = DiffusionPipeline.from_pretrained(
            #     "refiner-cache",
            #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
            #     vae=self.txt2img_pipe.vae,
            #     torch_dtype=torch.float16,
            #     use_safetensors=True,
            #     variant="fp16",
            # )
            # self.refiner.to("cuda")

        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")
        loaded_pose_image = None
        if pose_image:
            print("controlnet mode")
            loaded_image = self.load_image(pose_image)
            loaded_pose_image = loaded_image
            sdxl_kwargs["image"] = loaded_image

            # Get the dimensions (height and width) of the loaded image
            image_width, image_height = loaded_image.size

            sdxl_kwargs["target_size"] = (image_width, image_height)
            sdxl_kwargs["original_size"] = (image_width, image_height)
            pipe = self.controlnet_pipe_txt2img
        elif image and mask:
            print("inpainting mode")
            loaded_image = self.load_image(image)
            sdxl_kwargs["image"] = loaded_image
            sdxl_kwargs["mask_image"] = self.load_image(mask)
            sdxl_kwargs["strength"] = prompt_strength

            # Get the dimensions (height and width) of the loaded image
            image_width, image_height = loaded_image.size

            sdxl_kwargs["target_size"] = (image_width, image_height)
            sdxl_kwargs["original_size"] = (image_width, image_height)

            pipe = self.inpaint_pipe
        elif image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        # toggles watermark for this prediction
        pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs)

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }

            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        output_paths = []
        for i, _ in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        face_masks = face_mask_google_mediapipe(output.images, mask_blur_amount)

        # Based on face detection, crop base image, mask image and pose image (if available)
        # to the face and save them to output_paths
        (
            cropped_face,
            cropped_mask,
            cropped_control,
            left_top,
            orig_size,
        ) = crop_faces_to_square(
            output.images[0],
            face_masks[0],
            loaded_pose_image,
            face_padding,
            face_resize_to,
        )

        # Add face masks to output
        for i, _ in enumerate(face_masks):
            output_path = f"/tmp/out-{i}-mask.png"
            face_masks[i].save(output_path)
            output_paths.append(Path(output_path))

        # Add all cropped images to output
        output_path = f"/tmp/out-cropped-face.png"
        cropped_face.save(output_path)
        output_paths.append(Path(output_path))

        output_path = f"/tmp/out-cropped-mask.png"
        cropped_mask.save(output_path)
        output_paths.append(Path(output_path))

        if cropped_control:
            output_path = f"/tmp/out-cropped-control.png"
            cropped_control.save(output_path)
            output_paths.append(Path(output_path))

        upscaled_face = self.upscale_image_pil(cropped_face)

        inpaint_kwargs = {}

        inpaint_kwargs["prompt"] = inpaint_prompt
        inpaint_kwargs["negative_prompt"] = inpaint_negative_prompt
        inpaint_kwargs["image"] = upscaled_face
        inpaint_kwargs["mask_image"] = cropped_mask
        if cropped_control:
            inpaint_kwargs["control_image"] = cropped_control
        inpaint_kwargs["width"] = cropped_face.width
        inpaint_kwargs["height"] = cropped_face.height
        inpaint_kwargs["strength"] = inpaint_strength
        inpaint_kwargs["num_inference_steps"] = inpaint_num_inference_steps
        inpaint_kwargs["guidance_scale"] = inpaint_guidance_scale
        inpaint_kwargs["generator"] = torch.Generator("cuda").manual_seed(seed)

        if cropped_control:
            inpaint_result = self.controlnet_pipe_inpaint(
                **inpaint_kwargs,
            )
        else:
            inpaint_result = self.inpaint_pipe(
                **inpaint_kwargs,
            )

        pasted_image = paste_inpaint_into_original_image(
            output.images[0],
            cropped_mask,
            left_top,
            inpaint_result.images[0],
            orig_size,
        )

        # Add inpaint result to output
        output_path = f"/tmp/out-inpaint.png"
        inpaint_result.images[0].save(output_path)
        output_paths.append(Path(output_path))

        # Add pasted image to output
        output_path = f"/tmp/out-pasted.png"
        pasted_image.save(output_path)
        output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
