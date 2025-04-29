from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import cv2
from django.views.decorators.csrf import csrf_exempt

# Home page view
def home(request):
    return render(request, 'home.html')  # Render a simple home.html template

# Watermark removal view
@csrf_exempt
def remove_watermark(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']

        # Read image in OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Watermark removal
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # Encode and return image
        _, buffer = cv2.imencode('.jpg', result)
        return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
    
    return HttpResponse("Please POST an image file.", status=400)
