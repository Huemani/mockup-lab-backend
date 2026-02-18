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

# Pydantic models matching frontend
class Position(BaseModel):
    x: int
    y: int

class MockupRequest(BaseModel):
    baseImageUrl: str
    designImageUrl: str
    canvasWidth: int
    canvasHeight: int
    basePosition: Position
    baseScale: float
    position: Position
    scale: float
    rotation: float = 0.0           # degrees, -180 to 180
    perspectiveX: float = 0.0       # degrees, -45 to 45 (skew tilt)
    perspectiveY: float = 0.0       # degrees, -45 to 45 (skew lean)
    designNaturalWidth: int = 0     # design natural px width
    designNaturalHeight: int = 0    # design natural px height
    displacementStrength: int = 10
    shadowStrength: float = 0.5      # 0.0 to 1.0, shadow/highlight blend strength
    opacity: float = 1.0              # 0.0 to 1.0, design opacity
    method: str = "dtg"               # "dtg" or "embroidery"


def download_image_from_url(url: str, keep_alpha: bool = False):
    """Download image from URL and convert to OpenCV format"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    nparr = np.frombuffer(response.content, np.uint8)
    # Use IMREAD_UNCHANGED to preserve alpha channel when needed
    flag = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    img = cv2.imdecode(nparr, flag)
    return img


def create_displacement_map(tshirt_bgr):
    """
    Create a smooth displacement map from the t-shirt image.
    Replicates Photoshop approach:
    1. Convert to grayscale
    2. Apply strong Gaussian Blur to keep large folds, ignore fine texture
    3. Normalize to 0-255
    """
    gray = cv2.cvtColor(tshirt_bgr, cv2.COLOR_BGR2GRAY)
    # Blur radius controls how large the displacement features are.
    # Large radius = follows big folds only (more realistic).
    # Adjust based on image resolution — 51 works well for ~1000px images.
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    disp_map = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    # Re-center around 128 so the average displacement is zero.
    # Without this, images with non-neutral average brightness cause
    # the entire design to shift position globally.
    mean_val = np.mean(disp_map)
    disp_map = np.clip(disp_map - mean_val + 128, 0, 255).astype(np.uint8)

    return disp_map


def generate_contour_following_stitches(design_alpha, spacing=5):
    """
    Generate embroidery stitches that follow the contours of the design.
    This creates realistic directional stitching rather than uniform parallel lines.
    
    Args:
        design_alpha: Alpha channel of design (0-255)
        spacing: Spacing between stitch lines
        
    Returns:
        Stitch pattern image
    """
    h, w = design_alpha.shape
    stitch_pattern = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Find contours in the design
    _, binary = cv2.threshold(design_alpha, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"   Contour stitching: Found {len(contours)} contours")
    
    if len(contours) == 0:
        return stitch_pattern
    
    # 2. For each contour, fill with stitches that follow the shape
    for idx, contour in enumerate(contours):
        # Skip very small contours (noise)
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # Create mask for this contour
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Create temporary canvas for this contour's stitches
        temp_stitches = np.zeros((h, w), dtype=np.uint8)
        
        # Get bounding box
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Determine dominant direction for stitches
        # If wider than tall → horizontal stitches, else vertical
        if cw > ch * 1.2:
            # Horizontal stitches
            for i in range(y, y + ch, spacing):
                cv2.line(temp_stitches, (x, i), (x + cw, i), 255, 1)
        elif ch > cw * 1.2:
            # Vertical stitches
            for i in range(x, x + cw, spacing):
                cv2.line(temp_stitches, (i, y), (i, y + ch), 255, 1)
        else:
            # Diagonal stitches (45°)
            for offset in range(-max(cw, ch), max(cw, ch), spacing):
                x1 = x + max(0, -offset)
                y1 = y + max(0, offset)
                x2 = x + min(cw, cw - offset)
                y2 = y + min(ch, ch + offset)
                if x1 < x2 and y1 < y2:
                    cv2.line(temp_stitches, (x1, y1), (x2, y2), 255, 1)
        
        # Mask stitches to only inside this contour
        masked_stitches = cv2.bitwise_and(temp_stitches, contour_mask)
        
        # ACCUMULATE into main pattern (don't overwrite!)
        stitch_pattern = cv2.bitwise_or(stitch_pattern, masked_stitches)
    
    print(f"   Contour stitching: Generated {np.count_nonzero(stitch_pattern)} stitch pixels")
    
    return stitch_pattern


def generate_simple_stitch_pattern(design_alpha, angle=45, spacing=3):
    """
    Generate simplified embroidery stitch pattern.
    Creates parallel lines across design area at specified angle.
    
    Args:
        design_alpha: grayscale alpha channel (0-255)
        angle: stitch direction in degrees (0=horizontal, 45=diagonal, 90=vertical)
        spacing: pixels between stitch lines (default 3 for dense realistic look)
    
    Returns:
        Grayscale image with white stitch lines on black background
    """
    h, w = design_alpha.shape
    stitch_pattern = np.zeros((h, w), dtype=np.uint8)
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Determine how many lines we need to cover the entire canvas
    # Use diagonal length as safe upper bound
    diagonal = int(np.sqrt(h**2 + w**2))
    num_lines = diagonal // spacing + 1
    
    print(f"   Stitch generation: {w}x{h} canvas, angle={angle}°, spacing={spacing}px, drawing {num_lines*2} lines")
    
    # Draw parallel lines with thickness=2 for more visible stitches
    lines_drawn = 0
    for i in range(-num_lines, num_lines):
        offset = i * spacing
        
        # Calculate line endpoints perpendicular to stitch direction
        # Start from center and extend in perpendicular direction
        perp_dx = -dy
        perp_dy = dx
        
        # Line passes through (offset * perp_dx, offset * perp_dy) in direction (dx, dy)
        # Calculate endpoints that extend beyond canvas
        t_vals = [-diagonal, diagonal]
        
        for t1, t2 in [(t_vals[0], t_vals[1])]:
            x1 = int(w/2 + offset * perp_dx + t1 * dx)
            y1 = int(h/2 + offset * perp_dy + t1 * dy)
            x2 = int(w/2 + offset * perp_dx + t2 * dx)
            y2 = int(h/2 + offset * perp_dy + t2 * dy)
            
            # Draw THIN crisp line (thickness=1 for sharp SILK-like stitches)
            cv2.line(stitch_pattern, (x1, y1), (x2, y2), 255, 1)
            lines_drawn += 1
    
    print(f"   Stitch generation: Drew {lines_drawn} lines, pattern max={stitch_pattern.max()}, non-zero pixels before mask={np.count_nonzero(stitch_pattern)}")
    
    # Mask with design alpha - only keep stitches inside design
    stitch_pattern = cv2.bitwise_and(stitch_pattern, design_alpha)
    
    print(f"   Stitch generation: After alpha mask, non-zero pixels={np.count_nonzero(stitch_pattern)}, alpha has {np.count_nonzero(design_alpha)} non-zero pixels")
    
    return stitch_pattern


def generate_thread_texture(size=(256, 256)):
    """
    Generate procedural thread texture using Perlin-like noise.
    Creates vertical striations to simulate thread structure.
    
    Returns:
        Grayscale texture (0-255)
    """
    h, w = size
    texture = np.zeros((h, w), dtype=np.uint8)
    
    # Create vertical thread lines with slight variation
    for x in range(w):
        # Base intensity varies per column (thread)
        base = 128 + int(30 * np.sin(x * 0.5))
        
        for y in range(h):
            # Add vertical variation (twist in thread)
            variation = int(20 * np.sin(y * 0.1 + x * 0.3))
            intensity = np.clip(base + variation, 0, 255)
            texture[y, x] = intensity
    
    return texture


def apply_embroidery_effect(design_bgra, stitch_angle=45, stitch_spacing=4):
    """
    Apply advanced realistic embroidery effect.
    
    Features:
    1. Directional edge bevel (raised appearance)
    2. Uniform diagonal stitches  
    3. Subtle thread texture
    4. Color saturation boost
    
    Args:
        design_bgra: Design with alpha channel (BGRA)
        stitch_angle: Stitch direction (always 45°)
        stitch_spacing: Pixels between stitch lines
    
    Returns:
        Tuple of (embroidered_bgra, shadow_mask)
    """
    h, w = design_bgra.shape[:2]
    alpha = design_bgra[:, :, 3]
    design_bgr = design_bgra[:, :, :3].copy()
    
    # Calculate scale-adaptive blur ONLY (spacing stays tight at 2px always)
    scale_factor = w / 400.0
    adaptive_blur = max(5, int(11 * scale_factor))
    if adaptive_blur % 2 == 0:
        adaptive_blur += 1  # Must be odd for GaussianBlur
    
    print(f"  Embroidery: {w}x{h}, scale={scale_factor:.2f}, blur={adaptive_blur}px")
    
    # ==================================================================
    # STEP 1: DIRECTIONAL EDGE BEVEL (raised 3D appearance)
    # ==================================================================
    
    _, binary = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    if dist_transform.max() > 0:
        dist_transform = dist_transform / dist_transform.max()
    
    # OUTWARD-facing edge normals (negated to flip from inward)
    grad_x = -cv2.Sobel(dist_transform, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = -cv2.Sobel(dist_transform, cv2.CV_32F, 0, 1, ksize=5)
    
    # WIDER edge mask for smoother gradients
    edge_mask = (dist_transform > 0) & (dist_transform < 0.5)
    edge_mask_float = edge_mask.astype(np.float32)
    
    # Normalize gradients
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    norm_grad_x = grad_x / grad_magnitude
    norm_grad_y = grad_y / grad_magnitude
    
    # Light from top-right (45°)
    light_x, light_y = 0.707, -0.707
    light_facing = (norm_grad_x * light_x + norm_grad_y * light_y) * edge_mask_float
    
    # Split into highlight (faces light) and shadow (faces away)
    highlight_mask = np.clip(light_facing, 0, None)
    shadow_mask_edge = np.clip(-light_facing, 0, None)
    
    # Blur for smooth gradients (adaptive based on design size)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (adaptive_blur, adaptive_blur), 0)
    shadow_mask_edge = cv2.GaussianBlur(shadow_mask_edge, (adaptive_blur, adaptive_blur), 0)
    
    # Normalize
    if highlight_mask.max() > 0:
        highlight_mask = highlight_mask / highlight_mask.max()
    if shadow_mask_edge.max() > 0:
        shadow_mask_edge = shadow_mask_edge / shadow_mask_edge.max()
    
    # Apply bevel with SUBTLE strength to preserve original color
    embroidered_bgr = design_bgr.copy().astype(np.float32)
    embroidered_bgr -= shadow_mask_edge[:, :, np.newaxis] * 35
    embroidered_bgr += highlight_mask[:, :, np.newaxis] * 40
    embroidered_bgr = np.clip(embroidered_bgr, 0, 255).astype(np.uint8)
    
    print(f"  Embroidery: ✓ Smooth directional bevel")
    
    # ==================================================================
    # STEP 2: SHARP CRISP STITCHES (like SILK reference)
    # ==================================================================
    
    # Generate THIN stitches with FIXED 2px spacing (always tight)
    stitch_pattern = generate_simple_stitch_pattern(alpha, angle=45, spacing=2)
    
    embroidered_bgr = embroidered_bgr.astype(np.int16)
    
    # REDUCED highlights on stitch lines to preserve color
    embroidered_bgr[stitch_pattern > 0] += 35
    
    embroidered_bgr = np.clip(embroidered_bgr, 0, 255).astype(np.uint8)
    
    print(f"  Embroidery: ✓ Sharp crisp stitches")
    
    # ==================================================================
    # STEP 3: SUBTLE THREAD TEXTURE (NO BLUR - keep sharp)
    # ==================================================================
    
    # Very subtle noise for thread texture
    np.random.seed(42)
    fiber_noise = np.random.randint(-5, 5, (h, w), dtype=np.int16)
    
    # Apply only to stitch areas, NO BLUR
    stitch_mask = (stitch_pattern > 0)
    fiber_texture = fiber_noise * stitch_mask.astype(np.int16)
    
    embroidered_bgr = embroidered_bgr.astype(np.int16)
    embroidered_bgr += fiber_texture[:, :, np.newaxis]
    embroidered_bgr = np.clip(embroidered_bgr, 0, 255).astype(np.uint8)
    
    print(f"  Embroidery: ✓ Subtle sharp texture")
    
    # ==================================================================
    # STEP 4: SUBTLE SATURATION BOOST (preserve darkness)
    # ==================================================================
    
    # Convert to HSV and boost saturation SLIGHTLY
    embroidered_hsv = cv2.cvtColor(embroidered_bgr, cv2.COLOR_BGR2HSV)
    embroidered_hsv[:, :, 1] = np.clip(
        embroidered_hsv[:, :, 1].astype(np.float32) * 1.10,
        0, 255
    ).astype(np.uint8)
    embroidered_bgr = cv2.cvtColor(embroidered_hsv, cv2.COLOR_HSV2BGR)
    
    print(f"  Embroidery: ✓ Subtle saturation boost")
    
    # ==================================================================
    # STEP 5: PROJECTED SHADOW MASK
    # ==================================================================
    
    # Offset shadow (3px right, 4px down)
    shadow_projection = alpha.copy()
    M_shadow = np.float32([[1, 0, 3], [0, 1, 4]])
    shadow_projection = cv2.warpAffine(shadow_projection, M_shadow, (w, h))
    
    # Blur for soft shadow
    shadow_projection = cv2.GaussianBlur(shadow_projection, (13, 13), 0).astype(np.float32) / 255.0
    
    # Don't shadow where design itself is
    shadow_projection[alpha > 0] = 0
    
    print(f"  Embroidery: ✓ Projected shadow mask")
    
    # Combine with alpha
    embroidered = cv2.merge([
        embroidered_bgr[:, :, 0],
        embroidered_bgr[:, :, 1],
        embroidered_bgr[:, :, 2],
        alpha
    ])
    
    print(f"  Embroidery: Complete - all features applied")
    
    return embroidered, shadow_projection


def warp_design_with_displacement(design_bgra, disp_map, strength):
    """
    Warp the design image using the displacement map.
    Replicates Photoshop's Displace filter:
    - White pixels (255) shift design pixels in +x/+y direction
    - Black pixels (0) shift in -x/-y direction
    - Mid-gray (128) = no shift
    Strength corresponds to Photoshop's horizontal/vertical scale (default 10).
    """
    h, w = design_bgra.shape[:2]

    # Resize displacement map to match design dimensions
    disp_resized = cv2.resize(disp_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert grayscale values to pixel offsets
    # 128 = neutral (no shift), 0 = max negative, 255 = max positive
    disp_normalized = (disp_resized.astype(np.float32) - 128.0) / 128.0  # range: -1 to +1
    disp_pixels = disp_normalized * strength  # scale by strength in pixels

    # Blur the displacement FIELD (not just the map) to eliminate sharp gradients
    # that cause stretching artifacts at edges. This guarantees smooth transitions
    # between neighbouring pixels regardless of base image content.
    disp_pixels = cv2.GaussianBlur(disp_pixels, (15, 15), 0)

    # Build remap coordinates
    x_coords = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, axis=0)
    y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)

    map_x = x_coords + disp_pixels
    map_y = y_coords + disp_pixels

    # Apply warp to all 4 channels (BGRA) — critical to warp alpha too
    warped = cv2.remap(
        design_bgra,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # transparent border
    )

    return warped


def place_and_resize_design(tshirt_shape, design_bgra, pos_x, pos_y, design_width, design_height):
    """
    Resize design to target dimensions and place it on a transparent canvas
    the same size as the t-shirt image.
    Returns BGRA canvas.
    """
    th, tw = tshirt_shape[:2]
    canvas = np.zeros((th, tw, 4), dtype=np.uint8)

    dh, dw = design_bgra.shape[:2]
    # Maintain aspect ratio
    scale = min(design_width / dw, design_height / dh)
    new_w = int(dw * scale)
    new_h = int(dh * scale)

    # Use INTER_AREA for downscaling (avoids aliasing/pixelation),
    # INTER_CUBIC for upscaling (better quality than bilinear)
    interp = cv2.INTER_AREA if (new_w < dw or new_h < dh) else cv2.INTER_CUBIC
    resized = cv2.resize(design_bgra, (new_w, new_h), interpolation=interp)

    # pos_x/pos_y is the CENTER of the design in base image pixels
    x_off = pos_x - new_w // 2
    y_off = pos_y - new_h // 2

    # Clip to canvas bounds
    x_off = max(0, min(x_off, tw - 1))
    y_off = max(0, min(y_off, th - 1))
    x_end = min(x_off + new_w, tw)
    y_end = min(y_off + new_h, th)

    # Clip design if it extends beyond canvas
    design_x_end = new_w - max(0, (x_off + new_w) - tw)
    design_y_end = new_h - max(0, (y_off + new_h) - th)

    canvas[y_off:y_end, x_off:x_end] = resized[0:design_y_end, 0:design_x_end]
    return canvas


def alpha_composite_design(tshirt_bgr, warped_design_bgra, tshirt_bgr_original, shadow_strength=0.5, opacity=1.0):
    """
    Composite the warped design onto the t-shirt with realistic shadow/highlight integration.

    Steps:
    1. Alpha-composite warped design cleanly over t-shirt (RGB untouched)
    2. Convert composited result to LAB colour space
    3. Replace L channel with a blend of design L and t-shirt L
       — this adjusts brightness only, A and B (colour) channels are physically unchanged
    4. Convert back to BGR

    LAB is the only truly colour-neutral approach: L = pure lightness, A/B = pure colour.
    Shadows darken the design, highlights brighten it, zero colour contamination.
    """
    # Split warped design into BGR + alpha
    b, g, r, a = cv2.split(warped_design_bgra)
    design_bgr = cv2.merge([b, g, r])
    alpha_mask = a.astype(np.float32) / 255.0
    alpha_mask = alpha_mask * opacity  # apply opacity to alpha
    alpha_3ch = alpha_mask[:, :, np.newaxis]

    # Step 1: Clean alpha composite in RGB
    tshirt_float = tshirt_bgr.astype(np.float32) / 255.0
    design_float = design_bgr.astype(np.float32) / 255.0
    composited = design_float * alpha_3ch + tshirt_float * (1.0 - alpha_3ch)
    composited_uint8 = np.clip(composited * 255, 0, 255).astype(np.uint8)

    # Step 2: Convert both composited result and t-shirt to LAB
    composited_lab = cv2.cvtColor(composited_uint8, cv2.COLOR_BGR2Lab)
    tshirt_lab = cv2.cvtColor(tshirt_bgr, cv2.COLOR_BGR2Lab)

    # Step 3: Extract L channels (lightness only)
    comp_L = composited_lab[:, :, 0].astype(np.float32)   # design composited lightness
    tshirt_L = tshirt_lab[:, :, 0].astype(np.float32)     # fabric lightness (shadows/highlights)

    # Blend: pull design L toward fabric L, weighted by shadow_strength and alpha
    # At strength=0: pure design lightness (no fabric influence)
    # At strength=1: full fabric lightness mapped onto design
    # Fabric L is normalised: 128 = neutral (no shift), <128 = shadow, >128 = highlight
    fabric_influence = (tshirt_L - 128.0) * shadow_strength  # signed offset: negative=dark, positive=light
    blended_L = comp_L + fabric_influence * alpha_mask        # only affect where design exists
    blended_L = np.clip(blended_L, 0, 255).astype(np.uint8)

    # Step 4: Recombine with original A and B channels (colour untouched)
    result_lab = composited_lab.copy()
    result_lab[:, :, 0] = blended_L

    # Step 5: Convert back to BGR
    result = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    return result


@app.get("/")
async def root():
    return {
        "message": "Mockup Lab API",
        "status": "running",
        "version": "3.0.0",
        "endpoints": ["/health", "/generate-mockup", "/docs"],
        "accepts": "JSON with Cloudinary URLs (frontend contract)"
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
    Generate t-shirt mockup with Photoshop-equivalent displacement mapping.

    Pipeline:
    1. Download t-shirt (BGR) and design (BGRA with alpha)
    2. Create displacement map: grayscale t-shirt → strong Gaussian Blur
    3. Place design on canvas at correct position/scale
    4. Warp design using displacement map (Photoshop Displace filter equivalent)
    5. Alpha-composite warped design onto t-shirt
    6. Apply Multiply blend at low opacity for shadow integration
    7. Upload result to Cloudinary
    """
    try:
        print(f"\n{'='*60}")
        print("MOCKUP GENERATION REQUEST v3.0")
        print(f"{'='*60}")
        print(f"  Base Image URL:   {request.baseImageUrl}")
        print(f"  Design Image URL: {request.designImageUrl}")
        print(f"  Canvas:           {request.canvasWidth}x{request.canvasHeight}")
        print(f"  Position:         ({request.position.x}, {request.position.y})")
        print(f"  Scale:            {request.scale}")
        print(f"  Method:           {request.method.upper()}")
        print(f"  Displacement:     {request.displacementStrength}")
        print(f"  Shadow Strength:  {request.shadowStrength}")
        print(f"  Opacity:          {request.opacity}")

        # --- 1. Download images ---
        print("\n→ Downloading t-shirt image...")
        tshirt_bgr = download_image_from_url(request.baseImageUrl, keep_alpha=False)

        print("→ Downloading design image (with alpha)...")
        design_bgra = download_image_from_url(request.designImageUrl, keep_alpha=True)

        if tshirt_bgr is None or design_bgra is None:
            raise HTTPException(status_code=400, detail="Failed to download images from URLs")

        # Ensure design has alpha channel
        if design_bgra.ndim == 2:
            # Grayscale → BGRA
            design_bgra = cv2.cvtColor(design_bgra, cv2.COLOR_GRAY2BGR)
            design_bgra = cv2.cvtColor(design_bgra, cv2.COLOR_BGR2BGRA)
        elif design_bgra.shape[2] == 3:
            # BGR → BGRA (no transparency → fully opaque)
            design_bgra = cv2.cvtColor(design_bgra, cv2.COLOR_BGR2BGRA)

        print(f"✓ T-shirt loaded:  {tshirt_bgr.shape}")
        print(f"✓ Design loaded:   {design_bgra.shape} (channels: {design_bgra.shape[2]})")

        # --- 2. Create displacement map from t-shirt ---
        print("→ Creating displacement map...")
        disp_map = create_displacement_map(tshirt_bgr)
        print("✓ Displacement map created")

        # --- 3. Calculate design size in base image pixels ---
        # scaleInBase = design natural pixels * scale → base image pixels
        # e.g. design is 500px wide, scaleInBase=0.5 → 250px wide on base image
        if request.designNaturalWidth > 0 and request.designNaturalHeight > 0:
            design_width  = int(request.designNaturalWidth  * request.scale)
            design_height = int(request.designNaturalHeight * request.scale)
        else:
            # Fallback if frontend did not send natural dimensions
            design_width  = int(tshirt_bgr.shape[1] * request.scale * 0.3)
            design_height = int(tshirt_bgr.shape[0] * request.scale * 0.3)
        print(f"✓ Design size:     {design_width}x{design_height} at ({request.position.x},{request.position.y}) rot={request.rotation}°")

        # --- 4. Place design on full-size canvas (BGRA) ---
        print("→ Placing design on canvas...")
        design_canvas = place_and_resize_design(
            tshirt_bgr.shape,
            design_bgra,
            request.position.x,
            request.position.y,
            design_width,
            design_height
        )
        print("✓ Design placed")

        # --- 4b. Apply rotation around design center ---
        if request.rotation != 0.0:
            print(f"→ Rotating design {request.rotation}°...")
            th, tw = tshirt_bgr.shape[:2]
            cx, cy = request.position.x, request.position.y
            M = cv2.getRotationMatrix2D((cx, cy), -request.rotation, 1.0)
            design_canvas = cv2.warpAffine(
                design_canvas, M, (tw, th),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            print("✓ Rotation applied")

        # --- 4c. Apply skew (perspective) around design center ---
        # Replicates PixiJS: skew = (degrees * π / 180) * 0.1
        if request.perspectiveX != 0.0 or request.perspectiveY != 0.0:
            import math
            skew_x = (request.perspectiveX * math.pi / 180) * 0.1
            skew_y = (request.perspectiveY * math.pi / 180) * 0.1
            cx, cy = float(request.position.x), float(request.position.y)
            th, tw = tshirt_bgr.shape[:2]
            # Shear matrix around center point
            # x' = x + skew_x * (y - cy)
            # y' = skew_y * (x - cx) + y
            M_skew = np.float32([
                [1,      skew_x, -skew_x * cy],
                [skew_y, 1,      -skew_y * cx]
            ])
            design_canvas = cv2.warpAffine(
                design_canvas, M_skew, (tw, th),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            print(f"✓ Skew applied (x={skew_x:.4f}, y={skew_y:.4f} rad)")

        # --- 4d. Apply embroidery effect if method is embroidery ---
        shadow_mask = None  # Initialize shadow mask
        if request.method == "embroidery":
            print("→ Applying embroidery effect...")
            print(f"   Design canvas before embroidery: {design_canvas.shape}, has alpha: {design_canvas.shape[2] == 4}")
            design_canvas, shadow_mask = apply_embroidery_effect(
                design_canvas,
                stitch_angle=45,  # Will be auto-detected inside function
                stitch_spacing=2  # Very tight spacing for SILK-like crisp stitches
            )
            print("✓ Embroidery effect applied")
            print(f"   Design canvas after embroidery: {design_canvas.shape}")
            print(f"   Shadow mask shape: {shadow_mask.shape}")
        else:
            print(f"→ Skipping embroidery (method={request.method})")

        # --- 5. Warp design with displacement map ---
        print("→ Warping design with displacement map...")
        warped_design = warp_design_with_displacement(
            design_canvas,
            disp_map,
            request.displacementStrength
        )
        print("✓ Design warped")
        
        # --- 5b. Apply projected shadow to fabric (if embroidery) ---
        tshirt_with_shadow = tshirt_bgr.copy()
        if shadow_mask is not None and np.sum(shadow_mask) > 0:
            print("→ Applying projected shadow to fabric...")
            
            # Recreate warp coordinates from displacement map (same as design warping)
            h, w = shadow_mask.shape[:2]
            disp_resized = cv2.resize(disp_map, (w, h), interpolation=cv2.INTER_LINEAR)
            disp_normalized = (disp_resized.astype(np.float32) - 128.0) / 128.0
            disp_pixels = disp_normalized * request.displacementStrength
            disp_pixels = cv2.GaussianBlur(disp_pixels, (15, 15), 0)
            
            x_coords = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, axis=0)
            y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
            
            map_x = x_coords + disp_pixels
            map_y = y_coords + disp_pixels
            
            # Warp shadow mask
            warped_shadow = cv2.remap(
                shadow_mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Darken fabric where shadow falls
            shadow_darkening = (warped_shadow * 40).astype(np.int16)
            tshirt_with_shadow = tshirt_bgr.astype(np.int16)
            tshirt_with_shadow -= shadow_darkening[:, :, np.newaxis]
            tshirt_with_shadow = np.clip(tshirt_with_shadow, 0, 255).astype(np.uint8)
            print("✓ Projected shadow applied to fabric")
        else:
            print("→ No shadow mask to apply")

        # --- 6. Composite warped design onto t-shirt ---
        print("→ Compositing design onto t-shirt...")
        result = alpha_composite_design(tshirt_with_shadow, warped_design, tshirt_bgr, request.shadowStrength, request.opacity)
        print("✓ Composite complete")

        # --- 7. Upload to Cloudinary ---
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)

        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)

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

        return upload_result['secure_url']

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
    port = int(os.getenv("PORT", 8080))
    print(f"\nStarting Mockup Lab API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
