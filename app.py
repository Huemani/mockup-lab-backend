# app.py - Main backend application
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
from io import BytesIO
import os
from PIL import Image
from typing import Optional
import requests

# Initialize FastAPI
app = FastAPI(
    title="Mockup Lab API",
    docs_url="/docs",
    redoc_url=None
)

# CORS - Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load from environment variables
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

print("=" * 50)
print("CLOUDINARY CONFIGURATION CHECK:")
print(f"  Cloud name: {CLOUDINARY_CLOUD_NAME}")
print(f"  API key exists: {bool(CLOUDINARY_API_KEY)}")
print(f"  API secret exists: {bool(CLOUDINARY_API_SECRET)}")
print("=" * 50)

if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
    print("✓ Cloudinary configured successfully")
else:
    print("✗ Cloudinary configuration incomplete!")

# Pydantic model for JSON request
class MockupRequest(BaseModel):
    tshirt_url: str
    design_url: str
    strength: int = 10
    design_x: int = 0
    design_y: int = 0
    design_width: Optional[int] = None
    design_height: Optional[int] = None

def download_image_from_url(url: str):
    """Download image from URL and convert to OpenCV format"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def estimate_depth_simple(image):
    """
    Simple depth estimation from shading
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth = 255 - gray
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth

def create_displacement_from_best_channel(image):
    """
    Create displacement map by selecting channel with most contrast
    """
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    std_devs = [np.std(ch) for ch in channels]
    best_channel = channels[np.argmax(std_devs)]
    
    if len(best_channel.shape) == 3:
        gray = cv2.cvtColor(best_channel, cv2.COLOR_BGR2GRAY)
    else:
        gray = best_channel
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (21, 21), 0)
    
    return blurred

def compute_normals_from_depth(depth_map):
    """
    Compute surface normals from depth map
    """
    grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(grad_x)
    
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= magnitude
    normal_y /= magnitude
    normal_z /= magnitude
    
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    return normals

def create_physical_displacement_field(depth_map, normal_map, disp_map, strength=10):
    """
    Create physically-aware displacement field
    """
    h, w = depth_map.shape
    
    depth_norm = (depth_map.astype(float) - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    
    grad_x = cv2.Sobel(depth_norm, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=5)
    
    disp_x = grad_x * strength
    disp_y = grad_y * strength
    
    disp_norm = (disp_map.astype(float) / 255.0 - 0.5) * 2
    disp_x += disp_norm * (strength * 0.3)
    disp_y += disp_norm * (strength * 0.3)
    
    disp_x = cv2.GaussianBlur(disp_x, (15, 15), 0)
    disp_y = cv2.GaussianBlur(disp_y, (15, 15), 0)
    
    return disp_x.astype(np.float32), disp_y.astype(np.float32)

def apply_dense_warp(image, disp_x, disp_y):
    """
    Apply dense displacement field to warp image
    """
    h, w = image.shape[:2]
    
    x_coords = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
    y_coords = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
    
    map_x = x_coords + disp_x
    map_y = y_coords + disp_y
    
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
    if tshirt.shape[:2] != design.shape[:2]:
        design = cv2.resize(design, (tshirt.shape[1], tshirt.shape[0]))
    
    tshirt_float = tshirt.astype(float) / 255.0
    design_float = design.astype(float) / 255.0
    blended = tshirt_float * design_float
    
    if normal_map is not None:
        facing_factor = (normal_map[:,:,2] + 1) / 2
        facing_factor = np.clip(facing_factor, 0.3, 1.0)
        facing_factor = facing_factor[:,:,np.newaxis]
        adaptive_opacity = design_opacity * facing_factor
    else:
        adaptive_opacity = design_opacity
    
    result = blended * adaptive_opacity + tshirt_float * (1 - adaptive_opacity)
    
    tshirt_gray = cv2.cvtColor(tshirt, cv2.COLOR_BGR2GRAY)
    shadow_threshold = np.percentile(tshirt_gray, 15)
    shadow_mask = tshirt_gray < shadow_threshold
    shadow_mask = shadow_mask[:,:,np.newaxis]
    
    result = np.where(shadow_mask, 
                     result * 0.6 + tshirt_float * 0.4,
                     result)
    
    result = (result * 255).astype(np.uint8)
    return result

def resize_design_to_region(design, region_width, region_height):
    """
    Resize design to fit within region while maintaining aspect ratio
    """
    design_h, design_w = design.shape[:2]
    
    scale_w = region_width / design_w
    scale_h = region_height / design_h
    scale = min(scale_w, scale_h)
    
    new_w = int(design_w * scale)
    new_h = int(design_h * scale)
    
    resized = cv2.resize(design, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized

def place_design_on_tshirt(tshirt, design, x, y, width, height):
    """
    Place design on specific region of t-shirt
    """
    design_resized = resize_design_to_region(design, width, height)
    canvas = np.zeros_like(tshirt)
    
    dh, dw = design_resized.shape[:2]
    
    x_offset = x + (width - dw) // 2
    y_offset = y + (height - dh) // 2
    
    x_offset = max(0, min(x_offset, tshirt.shape[1] - dw))
    y_offset = max(0, min(y_offset, tshirt.shape[0] - dh))
    
    canvas[y_offset:y_offset+dh, x_offset:x_offset+dw] = design_resized
    return canvas

@app.get("/")
async def root():
    return {
        "message": "Mockup Lab API", 
        "status": "running",
        "version": "2.0.0",
        "endpoints": ["/health", "/generate-mockup", "/docs"],
        "accepts": "JSON with Cloudinary URLs"
    }

@app.get("/health")
async def health_check():
    cloudinary_configured = bool(
        CLOUDINARY_CLOUD_NAME and 
        CLOUDINARY_API_KEY and 
        CLOUDINARY_API_SECRET
    )
    return {
        "status": "healthy", 
        "cloudinary": cloudinary_configured,
        "cloud_name": CLOUDINARY_CLOUD_NAME or "not set",
        "api_key_set": bool(CLOUDINARY_API_KEY),
        "api_secret_set": bool(CLOUDINARY_API_SECRET)
    }

@app.post("/generate-mockup")
async def generate_mockup(request: MockupRequest):
    """
    Generate t-shirt mockup with optimal displacement mapping
    Accepts JSON with Cloudinary URLs
    """
    try:
        print(f"\n{'='*60}")
        print("MOCKUP GENERATION REQUEST (URL-based)")
        print(f"{'='*60}")
        print(f"  T-shirt URL: {request.tshirt_url}")
        print(f"  Design URL: {request.design_url}")
        print(f"  Strength: {request.strength}")
        
        # Download images from URLs
        print("→ Downloading t-shirt image...")
        tshirt_img = download_image_from_url(request.tshirt_url)
        
        print("→ Downloading design image...")
        design_img = download_image_from_url(request.design_url)
        
        if tshirt_img is None or design_img is None:
            raise HTTPException(status_code=400, detail="Failed to download images from URLs")
        
        print(f"✓ Images loaded: tshirt {tshirt_img.shape}, design {design_img.shape}")
        
        # Default design dimensions
        design_width = request.design_width or tshirt_img.shape[1] // 3
        design_height = request.design_height or tshirt_img.shape[0] // 3
        
        print(f"✓ Design placement: ({request.design_x}, {request.design_y}) size: {design_width}x{design_height}")
        
        # Place design
        design_placed = place_design_on_tshirt(
            tshirt_img, design_img, request.design_x, request.design_y, design_width, design_height
        )
        print("✓ Design placed on canvas")
        
        # Depth estimation
        depth = estimate_depth_simple(tshirt_img)
        print("✓ Depth estimated")
        
        # Compute normals
        normals = compute_normals_from_depth(depth)
        print("✓ Normals computed")
        
        # Displacement map
        disp_map = create_displacement_from_best_channel(tshirt_img)
        print("✓ Displacement map created")
        
        # Physical displacement field
        disp_x, disp_y = create_physical_displacement_field(
            depth, normals, disp_map, request.strength
        )
        print("✓ Displacement field generated")
        
        # Warp design
        warped_design = apply_dense_warp(design_placed, disp_x, disp_y)
        print("✓ Design warped")
        
        # Intelligent blend
        result = intelligent_blend(tshirt_img, warped_design, normals, depth)
        print("✓ Blending complete")
        
        # Convert to PIL Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Save to BytesIO
        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Upload to Cloudinary
        print("→ Uploading to Cloudinary...")
        
        if not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
            raise HTTPException(status_code=500, detail="Cloudinary not configured")
        
        upload_result = cloudinary.uploader.upload(
            buffer,
            folder="mockup-lab",
            resource_type="image"
        )
        
        print(f"✓ Upload successful: {upload_result['secure_url']}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "mockup_url": upload_result['secure_url'],
            "public_id": upload_result['public_id']
        }
        
    except requests.RequestException as e:
        print(f"\n✗ ERROR downloading images: {str(e)}\n")
        raise HTTPException(status_code=400, detail=f"Failed to download images: {str(e)}")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\nStarting Mockup Lab API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
