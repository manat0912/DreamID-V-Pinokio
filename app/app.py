import os
import sys
import torch
import random
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from datetime import datetime
# import spaces # Removed for local execution

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dreamidv_wan
from dreamidv_wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from dreamidv_wan.utils.utils import cache_video

# Model Paths
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Wan", "Wan2.1-T2V-1.3B")
DREAMIDV_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DreamID-V", "dreamidv.pth")
DREAMIDV_FASTER_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DreamID-V", "dreamidv_faster.pth")

# Output Directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DreamIDVApp:
    def __init__(self):
        self.pipeline = None
        self.current_model_type = None

    def load_model(self, model_version="Standard"):
        if self.pipeline is not None and self.current_model_type == model_version:
            return

        ckpt = DREAMIDV_CKPT if model_version == "Standard" else DREAMIDV_FASTER_CKPT
        
        if model_version == "Standard":
            import dreamidv_wan
            cfg = dreamidv_wan.configs.swapface
            self.pipeline = dreamidv_wan.DreamIDV(
                config=cfg,
                checkpoint_dir=CKPT_DIR,
                dreamidv_ckpt=ckpt,
                device_id=0,
                offload_model=True
            )
        else:
            import dreamidv_wan_faster
            cfg = dreamidv_wan_faster.configs.swapface
            self.pipeline = dreamidv_wan_faster.DreamIDV(
                config=cfg,
                checkpoint_dir=CKPT_DIR,
                dreamidv_ckpt=ckpt,
                device_id=0,
                offload_model=True
            )
        
        self.current_model_type = model_version

    def generate_pose_and_mask(self, video_path, image_path, model_version):
        if model_version == "Standard":
            from generate_dreamidv import generate_pose_and_mask_videos
            return generate_pose_and_mask_videos(video_path, image_path)
        else:
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose'))
            from pose.extract import process_dwpose
            
            temp_dir = os.path.join(os.path.dirname(video_path), 'temp_generated')
            os.makedirs(temp_dir, exist_ok=True)
            video_name_base = os.path.basename(video_path).split('.')[0]
            pose_path = os.path.join(temp_dir, video_name_base + '_pose.mp4')
            mask_path = os.path.join(temp_dir, video_name_base + '_mask.mp4')
            
            process_dwpose(video_path, pose_path, mask_path)
            return pose_path, mask_path

    # @spaces.GPU # Removed for local execution
    def swap_face(self, ref_image, ref_video, model_version, size, steps, seed):
        self.load_model(model_version)
        
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        print(f"Generating with seed: {seed}")
        
        # Preprocess pose and mask
        pose_path, mask_path = self.generate_pose_and_mask(ref_video, ref_image, model_version)
        
        ref_paths = [ref_video, mask_path, ref_image]
        if model_version == "Standard":
            ref_paths.append(pose_path)
        
        video = self.pipeline.generate(
            "change face",
            ref_paths,
            size=SIZE_CONFIGS[size],
            sampling_steps=steps,
            seed=seed,
            offload_model=True
        )
        
        output_file = os.path.join(OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        cache_video(
            tensor=video[None],
            save_file=output_file,
            fps=24,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        return output_file

app_logic = DreamIDVApp()

def run_gradio():
    with gr.Blocks(title="DreamID-V Face Swap") as demo:
        gr.Markdown("# 🎭 DreamID-V Face Swap")
        gr.Markdown("High-fidelity video face swapping powered by Diffusion Transformer.")
        
        with gr.Row():
            with gr.Column():
                ref_image = gr.Image(type="filepath", label="Reference Face Image")
                ref_video = gr.Video(label="Driving Video")
                
                with gr.Accordion("Advanced Settings", open=False):
                    model_version = gr.Dropdown(["Standard", "Faster"], value="Standard", label="Model Version")
                    size = gr.Dropdown(list(SIZE_CONFIGS.keys()), value="1280*720", label="Resolution")
                    steps = gr.Slider(minimum=1, maximum=100, step=1, value=12, label="Steps")
                    seed = gr.Number(value=-1, label="Seed (-1 for random)")
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="Result Video")

        generate_btn.click(
            fn=app_logic.swap_face,
            inputs=[ref_image, ref_video, model_version, size, steps, seed],
            outputs=output_video
        )
        
    demo.launch(share=False)

if __name__ == "__main__":
    run_gradio()
