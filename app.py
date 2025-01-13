import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import torch
from depth_pipeline import Depth2ImgPipeline  # Assuming your class code is saved as depth_pipeline.py
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, DPTImageProcessor, DPTForDepthEstimation

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the pre-trained pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = Depth2ImgPipeline(
    AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='vae').to(device),
    CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='tokenizer'),
    CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='text_encoder').to(device),
    UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='unet').to(device),
    PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000),
    DPTImageProcessor.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='feature_extractor'),
    DPTForDepthEstimation.from_pretrained('stabilityai/stable-diffusion-2-depth', subfolder='depth_estimator')
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def index():
    return render_template('sketch-to-art.html')

@app.route('/upload', methods=['POST'])
def upload_and_generate():
    if 'sketch' not in request.files:
        return "No file part", 400

    file = request.files['sketch']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded sketch
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    # Generate art from sketch
    input_image = Image.open(input_path).convert('RGB')

    # Convert PIL image to NumPy array
    input_image = np.array(input_image)

    # Convert NumPy array to PyTorch tensor
    input_tensor = torch.from_numpy(input_image).float()

    # Normalize the image tensor to range [0, 1]
    input_tensor = input_tensor / 255.0

    # Change the dimensions to [C, H, W] and add batch dimension: [B, C, H, W]
    input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # Get the prompt and strength from the form
    prompt = request.form.get('prompt', 'A beautiful piece of art')
    strength = float(request.form.get('strength', 0.8))

    # Generate the image using the pipeline
    generated_images = pipeline(
        prompt=prompt,
        img=input_tensor,
        strength=strength,
        num_inference_steps=50,
        guidance_scale=7.5
    )

    # Save the generated image
    output_path = os.path.join(app.config['RESULT_FOLDER'], f"generated_{file.filename}")
    generated_images[0].save(output_path)

    return send_from_directory(app.config['RESULT_FOLDER'], f"generated_{file.filename}")

if __name__ == '__main__':
    app.run(debug=True)
