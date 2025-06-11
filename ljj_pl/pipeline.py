import os
import sys
import argparse
import torch
import cv2
import numpy as np
import argparse

# 解析命令行参数
args = argparse.ArgumentParser()
args.add_argument("--q_bits", type=int, default=8)
args.add_argument("--q_dim", type=str, default="per-head+per-dim")
args = args.parse_args()

# 添加项目根目录到Python路径
INFINITY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if INFINITY_ROOT not in sys.path:
    sys.path.append(INFINITY_ROOT)

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_img,
)


class InfinityPipeline:
    def __init__(self, model_type="infinity_2b", q_bits=8, q_dim="per-head+per-dim"):
        self.model_type = model_type
        self.q_bits = q_bits
        self.q_dim = q_dim
        self.args = self._get_default_args()
        self._load_models()
        
    def _get_default_args(self):
        base_args = {
            "pn": "1M",
            "use_scale_schedule_embedding": 0,
            "use_bit_label": 1,
            "cfg_insertion_layer": 0,
            "rope2d_normalized_by_hw": 2,
            "add_lvl_embeding_only_first_block": 1,
            "rope2d_each_sa_layer": 1,
            "text_encoder_ckpt": "/data/boxunxu/Infinity/flan-t5-xl",
            "text_channels": 2048,
            "cache_dir": '/dev/shm',
            "h_div_w_template": 1.000,
            "enable_model_cache":0,
            "use_flex_attn":0,
            "q_bits": self.q_bits,
            "q_dim": self.q_dim,
        }
        if self.model_type == "infinity_2b":
            model_specific_args = {
                "model_type": "infinity_2b",
                "checkpoint_type": "torch",
                "model_path": "/data/boxunxu/Infinity/infinity_2b_reg.pth",
                "vae_type": 32,
                "vae_path": "/data/boxunxu/Infinity/infinity_vae_d32reg.pth",
                "apply_spatial_patchify": 0,
                "bf16": 1,
            }
        elif self.model_type == "infinity_8b":
            model_specific_args = {
                "model_type": "infinity_8b",
                "checkpoint_type": "torch_shard",
                "model_path": "/data/jiaji_lu/Infinity/infinity_8b_weights",
                "vae_type": 14,
                "vae_path": "/data/jiaji_lu/Infinity/infinity_vae_d56_f8_14_patchify.pth",
                "apply_spatial_patchify": 1,
                "bf16": 1,
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        args = {**base_args, **model_specific_args}
        return argparse.Namespace(**args)

    def _load_models(self): 
        print(f"[Loading {self.model_type}]")
        
        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=self.args.text_encoder_ckpt
        )
        self.vae = load_visual_tokenizer(self.args)
        self.infinity = load_transformer(self.vae, self.args)
        
    def generate(
        self,
        prompt: str,
        guidance_scale: float = 3.0,
        tau: float = 1.0,  #set the attention softmax temperature.higher temperature means more randomness. 
                           #While lower temperature means more deterministic.
        seed: int = None,
        n_samples: int = 1,
        h_div_w: float = 1.0,
        out_dir: str = None,
        sampling_per_bits: int = 1,  # set the sampling per bits, higher means more details.
                                     # can be 1,2,4,8,16
    ):
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if out_dir is None:
            out_dir = f"output/infinity_{self.model_type}_evaluation/images"
        
        h_div_w_template_ = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn][
            "scales"
        ]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        with torch.cuda.amp.autocast():  # use mixed precision
            generated_images = []
            for _ in range(n_samples):
                img = gen_one_img(
                    self.infinity,
                    self.vae,
                    self.text_tokenizer,
                    self.text_encoder,
                    prompt,
                    g_seed=seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=guidance_scale,
                    tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[self.args.cfg_insertion_layer],
                    vae_type=self.args.vae_type,
                    sampling_per_bits=sampling_per_bits,
                    enable_positive_prompt=0,
                )
                generated_images.append(img)

        if n_samples > 1:
            final_image = np.concatenate(generated_images, axis=1)
        else:
            final_image = generated_images[0]

        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{prompt[:30]}_{seed}.png")
        cv2.imwrite(output_path, final_image.cpu().numpy())
        
        return final_image, output_path

def main():
    for model_type in ["infinity_2b", "infinity_8b"]:
        print(f"\nTesting {model_type}")
        pipeline = InfinityPipeline(
            model_type=model_type,
            q_bits=args.q_bits,
            q_dim=args.q_dim
        )
        
        test_prompts = [
            " ",
            #"Two young Japanese goth cosplay girls in fishnets, corsets, chokers, and black and white makeup with full body tattoos and intricate painted details.",
        ]
        
        for prompt in test_prompts:
            image, path = pipeline.generate(
                prompt=prompt,
                guidance_scale=3.0,  # set the guidance scale
                tau=1.0,            # set the attention softmax temperature
                n_samples=1,
                out_dir=f"pipeline_outputs/{model_type}/{args.q_bits}",
                sampling_per_bits=1,
            )
            print(f"Generated image saved to: {path}")

if __name__ == "__main__":
    #CUDA_VISIBLE_DEVICES=1 python pipeline.py --q_bits 8 --q_dim per-head+per-dim
    main()