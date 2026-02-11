# app.py - Main backend application
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
from io import BytesIO
import os
from PIL import Image
import torch
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# CORS - Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cloudinary configuration (from environment variables)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Optional: MiDaS model for depth estimation (loaded lazily)
midas_model = None
midas_transform = None

def load_midas():
    """Load MiDaS depth estimation model (only when needed)"""
    global midas_model, midas_transform
    if midas_model is None:
        print("Loading MiDaS model...")
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_model.eval()
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        print("MiDaS model loaded!")
    return midas_model, midas_transform

def read_image_file(file_bytes):
    """Convert uploaded file bytes to OpenCV image"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def estimate_depth_simple(image):
    """
    Simple depth estimation from shading
    Faster alternative to MiDaS
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert: darker = further away (in folds)
    depth = 255 - gray
    
    # Smooth to reduce noise
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    
    # Enhance contrast
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    
    return depth

def estimate_depth_midas(image):
    """
    High-quality depth estimation using MiDaS
    """
    model, transform = load_midas()
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    input_batch = transform(img_rgb)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_batch)
        depth = prediction.squeeze().cpu().numpy()
    
    # Resize to original dimensions
    depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
    
    # Normalize to 0-255
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return depth

def create_displacement_from_best_channel(image):
    """
    Create displacement map by selecting channel with most contrast
    """
    # Split channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    
    # Find channel with highest standard deviation (most contrast)
    std_devs = [np.std(ch) for ch in channels]
    best_channel = channels[np.argmax(std_devs)]
    
    # Convert to grayscale if needed
    if len(best_channel.shape) == 3:
        gray = cv2.cvtColor(best_channel, cv2.COLOR_BGR2GRAY)
    else:
        gray = best_channel
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur for smooth displacement
    blurred = cv2.GaussianBlur(enhanced, (21, 21), 0)
    
    return blurred

def compute_normals_from_depth(depth_map):
    """
    Compute surface normals from depth map
    """
    # Compute gradients
    grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    
    # Create normal vectors
    # Normal = (-grad_x, -grad_y, 1)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(grad_x)
    
    # Normalize vectors
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= magnitude
    normal_y /= magnitude
    normal_z /= magnitude
    
    # Stack into single array [H, W, 3]
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    
    return normals

def create_physical_displacement_field(depth_map, normal_map, disp_map, strength=10):
    """
    Create physically-aware displacement field
    """
    h, w = depth_map.shape
    
    # Normalize depth to 0-1
    depth_norm = (depth_map.astype(float) - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    
    # Compute depth gradients (how fast depth changes)
    grad_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=5)
    
    # Displacement based on depth gradient
    # Steep gradient = more warping
    disp_x = grad_x * strength
    disp_y = grad_y * strength
    
    # Combine with original displacement map for fine details
    disp_norm = (disp_map.astype(float) / 255.0 - 0.5) * 2  # -1 to 1
    disp_x += disp_norm * (strength * 0.3)
    disp_y += disp_norm * (strength * 0.3)
    
    # Smooth displacement field for natural look
    disp_x = cv2.GaussianBlur(disp_x, (15, 15), 0)
    disp_y = cv2.GaussianBlur(disp_y, (15, 15), 0)
    
    return disp_x.astype(np.float32), disp_y.astype(np.float32)

def apply_dense_warp(image, disp_x, disp_y):
    """
    Apply dense displacement field to warp image
    """
    h, w = image.shape[:2]
    
    # Create coordinate grid
    x_coords = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
    y_coords = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
    
    # Apply displacement
    map_x = x_coords + disp_x
    map_y = y_coords + disp_y
    
    # Warp image using cv2.remap
    warped = cv2.remap(
        image, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return warped

def intelligent_blend(tshirt, design, normal_map, depth_map, design_opacity=0.85):
    """
    Intelligent blending of design with t-shirt
    """
    # Ensure same dimensions
    if tshirt.shape[:2] != design.shape[:2]:
        design = cv2.resize(design, (tshirt.shape[1], tshirt.shape[0]))
    
    # 1. Multiply blend mode
    tshirt_float = tshirt.astype(float) / 255.0
    design_float = design.astype(float) / 255.0
    blended = tshirt_float * design_float
    
    # 2. Adaptive opacity based on surface angle (if normals available)
    if normal_map is not None:
        # Z-component of normal (facing towards camera)
        facing_factor = (normal_map[:,:,2] + 1) / 2  # Normalize to 0-1
        facing_factor = np.clip(facing_factor, 0.3, 1.0)  # Minimum 30% opacity
        facing_factor = facing_factor[:,:,np.newaxis]  # Add channel dimension
        
        adaptive_opacity = design_opacity * facing_factor
    else:
        adaptive_opacity = design_opacity
    
    # 3. Final composite
    result = blended * adaptive_opacity + tshirt_float * (1 - adaptive_opacity)
    
    # 4. Preserve deep shadows (darkest 15% of original)
    tshirt_gray = cv2.cvtColor(tshirt, cv2.COLOR_BGR2GRAY)
    shadow_threshold = np.percentile(tshirt_gray, 15)
    shadow_mask = tshirt_gray < shadow_threshold
    shadow_mask = shadow_mask[:,:,np.newaxis]  # Add channel dimension
    
    result = np.where(shadow_mask, 
                     result * 0.6 + tshirt_float * 0.4,  # More tshirt in shadows
                     result)
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)
    
    return result

def resize_design_to_region(design, region_width, region_height):
    """
    Resize design to fit within region while maintaining aspect ratio
    """
    design_h, design_w = design.shape[:2]
    
    # Calculate scale to fit within region
    scale_w = region_width / design_w
    scale_h = region_height / design_h
    scale = min(scale_w, scale_h)
    
    # New dimensions
    new_w = int(design_w * scale)
    new_h = int(design_h * scale)
    
    # Resize
    resized = cv2.resize(design, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return resized

def place_design_on_tshirt(tshirt, design, x, y, width, height):
    """
    Place design on specific region of t-shirt
    """
    # Resize design to fit region
    design_resized = resize_design_to_region(design, width, height)
    
    # Create a canvas same size as t-shirt with design placed
    canvas = np.zeros_like(tshirt)
    
    dh, dw = design_resized.shape[:2]
    
    # Center design in specified region
    x_offset = x + (width - dw) // 2
    y_offset = y + (height - dh) // 2
    
    # Ensure within bounds
    x_offset = max(0, min(x_offset, tshirt.shape[1] - dw))
    y_offset = max(0, min(y_offset, tshirt.shape[0] - dh))
    
    # Place design
    canvas[y_offset:y_offset+dh, x_offset:x_offset+dw] = design_resized
    
    return canvas

@app.get("/")
async def root():
    return {
        "message": "Mockup Lab API", 
        "status": "running",
        "endpoints": ["/health", "/generate-mockup"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "cloudinary": bool(os.getenv("CLOUDINARY_CLOUD_NAME"))}

@app.post("/generate-mockup")
async def generate_mockup(
    tshirt: UploadFile = File(...),
    design: UploadFile = File(...),
    strength: int = Form(10),
    use_midas: bool = Form(False),
    design_x: int = Form(0),
    design_y: int = Form(0),
    design_width: Optional[int] = Form(None),
    design_height: Optional[int] = Form(None),
):
    """
    Generate t-shirt mockup with optimal displacement mapping
    
    Parameters:
    - tshirt: T-shirt photo
    - design: Design/logo to place on t-shirt
    - strength: Displacement strength (5-20, default 10)
    - use_midas: Use MiDaS for high-quality depth (slower)
    - design_x, design_y: Position of design (pixels from top-left)
    - design_width, design_height: Size of design region
    """
    try:
        # Read uploaded images
        tshirt_bytes = await tshirt.read()
        design_bytes = await design.read()
        
        tshirt_img = read_image_file(tshirt_bytes)
        design_img = read_image_file(design_bytes)
        
        if tshirt_img is None or design_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        print(f"Processing mockup: tshirt {tshirt_img.shape}, design {design_img.shape}")
        
        # If no design dimensions specified, use 1/3 of t-shirt width
        if design_width is None:
            design_width = tshirt_img.shape[1] // 3
        if design_height is None:
            design_height = tshirt_img.shape[0] // 3
        
        # Place design on t-shirt region
        design_placed = place_design_on_tshirt(
            tshirt_img, 
            design_img, 
            design_x, 
            design_y, 
            design_width, 
            design_height
        )
        
        # Step 1: Depth estimation
        print("Estimating depth...")
        if use_midas:
            depth = estimate_depth_midas(tshirt_img)
        else:
            depth = estimate_depth_simple(tshirt_img)
        
        # Step 2: Compute normals
        print("Computing normals...")
        normals = compute_normals_from_depth(depth)
        
        # Step 3: Create displacement map
        print("Creating displacement map...")
        disp_map = create_displacement_from_best_channel(tshirt_img)
        
        # Step 4: Physical displacement field
        print("Creating displacement field...")
        disp_x, disp_y = create_physical_displacement_field(
            depth, normals, disp_map, strength
        )
        
        # Step 5: Warp design
        print("Warping design...")
        warped_design = apply_dense_warp(design_placed, disp_x, disp_y)
        
        # Step 6: Intelligent blend
        print("Blending...")
        result = intelligent_blend(tshirt_img, warped_design, normals, depth)
        
        # Convert to PIL Image for upload
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Save to BytesIO
        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Upload to Cloudinary
        print("Uploading to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            buffer,
            folder="mockup-lab",
            resource_type="image"
        )
        
        print(f"Mockup generated: {upload_result['secure_url']}")
        
        return {
            "success": True,
            "mockup_url": upload_result['secure_url'],
            "public_id": upload_result['public_id']
        }
        
    except Exception as e:
        print(f"Error generating mockup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
