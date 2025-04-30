from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import cv2
import requests
from skimage.restoration import inpaint

# Constants for watermark properties
WATERMARK_TEXT_DEFAULT = "Sample Watermark"
WATERMARK_FONT = cv2.FONT_HERSHEY_SIMPLEX
WATERMARK_FONT_SCALE = 2
WATERMARK_COLOR = (255, 165, 0)  # Orange (BGR format)
WATERMARK_THICKNESS = 5
WATERMARK_OPACITY = 0.7
WATERMARK_SCALE_FACTOR = 0.1
WATERMARK_PADDING = 20
REQUEST_TIMEOUT = 10  # Seconds
MAX_IMAGE_DIM = 1000  # Resize large images for performance

def _load_image_from_url(url):
    """Fetch and decode an image from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200 or not response.headers.get('content-type', '').startswith('image/'):
            raise ValueError("Invalid image URL")
        file_bytes = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from URL")
        return img
    except Exception as e:
        raise ValueError(f"Error loading image from URL: {str(e)}")

def _load_image_from_file(uploaded_file):
    """Read and decode an image from an uploaded file."""
    if not uploaded_file.content_type.startswith('image/'):
        raise ValueError("Uploaded file is not an image")
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode uploaded image")
    return img

def _load_watermark_image(uploaded_file):
    """Read and decode a watermark image, preserving alpha channel."""
    if not uploaded_file.content_type.startswith('image/'):
        raise ValueError("Watermark image is not a valid image file")
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode watermark image")
    return img

def _encode_image_to_response(img):
    """Encode an image to JPEG and return as an HTTP response."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return HttpResponse(buffer.tobytes(), content_type='image/jpeg')

def _resize_image(img, max_dim=MAX_IMAGE_DIM):
    """Resize image if it exceeds max dimension for faster processing."""
    height, width = img.shape[:2]
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def _enhanced_remove_watermark(img):
    """
    Remove watermarks using a simplified, efficient pipeline:
    1. Convert to grayscale and HSV for detection.
    2. Detect watermarks using HSV color ranges and edge detection.
    3. Inpaint detected regions using OpenCV and skimage.
    """
    # Ensure image is RGB
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Resize for performance
    img = _resize_image(img)
    
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect common watermark colors in HSV (orange, white, light blue)
    color_ranges = [
        (np.array([10, 100, 100]), np.array([35, 255, 255])),  # Orange
        (np.array([0, 0, 180]), np.array([180, 30, 255])),     # White/light gray
        (np.array([90, 50, 180]), np.array([130, 255, 255]))   # Light blue
    ]
    
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for lower, upper in color_ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    # Detect edges for semi-transparent watermarks
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    mask |= edge_mask

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint using OpenCV (fast)
    cv2_result = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Inpaint using skimage (better quality for complex areas)
    img_float = img.astype(np.float32) / 255.0
    skimage_result = inpaint.inpaint_biharmonic(
        img_float,
        mask.astype(bool),
        channel_axis=-1
    )
    skimage_result = (skimage_result * 255).astype(np.uint8)

    # Blend results for best quality
    return cv2.addWeighted(skimage_result, 0.6, cv2_result, 0.4, 0)

def _add_text_watermark(img, text):
    """Add a semi-transparent text watermark to the center of the image."""
    height, width = img.shape[:2]
    text_size = cv2.getTextSize(text, WATERMARK_FONT, WATERMARK_FONT_SCALE, WATERMARK_THICKNESS)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    overlay = img.copy()
    cv2.putText(
        overlay, text, (text_x, text_y),
        WATERMARK_FONT, WATERMARK_FONT_SCALE,
        WATERMARK_COLOR, WATERMARK_THICKNESS
    )
    cv2.addWeighted(overlay, WATERMARK_OPACITY, img, 1 - WATERMARK_OPACITY, 0, img)

def _add_image_watermark(img, watermark_img):
    """Add a watermark image to the bottom right corner."""
    height, width = img.shape[:2]
    wm_height, wm_width = watermark_img.shape[:2]

    # Resize watermark
    scale = (width * WATERMARK_SCALE_FACTOR) / wm_width
    new_wm_size = (int(wm_width * scale), int(wm_height * scale))
    watermark_img = cv2.resize(watermark_img, new_wm_size, interpolation=cv2.INTER_AREA)

    wm_height, wm_width = watermark_img.shape[:2]
    wm_x = width - wm_width - WATERMARK_PADDING
    wm_y = height - wm_height - WATERMARK_PADDING

    if wm_x < 0 or wm_y < 0:
        raise ValueError("Watermark image too large for the main image")

    # Handle alpha channel if present
    if watermark_img.shape[2] == 4:
        wm_rgb = watermark_img[:, :, :3]
        wm_alpha = watermark_img[:, :, 3] / 255.0
        roi = img[wm_y:wm_y + wm_height, wm_x:wm_x + wm_width]
        for c in range(3):
            roi[:, :, c] = (1 - wm_alpha) * roi[:, :, c] + wm_alpha * wm_rgb[:, :, c]
    else:
        overlay = img.copy()
        overlay[wm_y:wm_y + wm_height, wm_x:wm_x + wm_width] = watermark_img
        cv2.addWeighted(overlay, WATERMARK_OPACITY, img, 1 - WATERMARK_OPACITY, 0, img)

def home(request):
    """Render the home page."""
    return render(request, 'home.html')

def remove_watermark(request):
    """Remove watermarks from an image via URL (GET) or upload (POST)."""
    if request.method == 'GET':
        image_url = request.GET.get('image_url')
        if not image_url:
            return render(request, 'home.html')
        try:
            img = _load_image_from_url(image_url)
            result = _enhanced_remove_watermark(img)
            return _encode_image_to_response(result)
        except ValueError as e:
            return HttpResponse(str(e), status=400)
        except Exception as e:
            return HttpResponse(f"Error processing image: {str(e)}", status=500)

    elif request.method == 'POST' and request.FILES.get('image'):
        try:
            img = _load_image_from_file(request.FILES['image'])
            result = _enhanced_remove_watermark(img)
            return _encode_image_to_response(result)
        except ValueError as e:
            return HttpResponse(str(e), status=400)
        except Exception as e:
            return HttpResponse(f"Error processing image: {str(e)}", status=500)

    return HttpResponse("Invalid request. Use GET with image_url or POST to upload an image.", status=400)

def add_watermark(request):
    """Add watermarks (text and/or image) to an image via upload (POST) or URL (GET)."""
    if request.method == 'POST':
        if not request.FILES.get('main_image'):
            return HttpResponse("Please upload a main image", status=400)
        try:
            img = _load_image_from_file(request.FILES['main_image'])
            watermark_text = request.POST.get('text', WATERMARK_TEXT_DEFAULT)
            _add_text_watermark(img, watermark_text)
            if request.FILES.get('watermark_image'):
                watermark_img = _load_watermark_image(request.FILES['watermark_image'])
                _add_image_watermark(img, watermark_img)
            return _encode_image_to_response(img)
        except ValueError as e:
            return HttpResponse(str(e), status=400)
        except Exception as e:
            return HttpResponse(f"Error adding watermark: {str(e)}", status=500)

    elif request.method == 'GET':
        image_url = request.GET.get('image_url')
        watermark_url = request.GET.get('watermark_url')
        watermark_text = request.GET.get('text', WATERMARK_TEXT_DEFAULT)
        if not image_url:
            return HttpResponse("Please provide an image_url parameter", status=400)
        try:
            img = _load_image_from_url(image_url)
            _add_text_watermark(img, watermark_text)
            if watermark_url:
                watermark_img = _load_image_from_url(watermark_url)
                _add_image_watermark(img, watermark_img)
            return _encode_image_to_response(img)
        except ValueError as e:
            return HttpResponse(str(e), status=400)
        except Exception as e:
            return HttpResponse(f"Error adding watermark: {str(e)}", status=500)

    return HttpResponse("Invalid request. Use POST with file uploads or GET with image_url.", status=400)