from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views  # Assuming views.py is in the same app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('remove_watermark/', views.remove_watermark, name='remove_watermark'),
    path('add_watermark/', views.add_watermark, name='add_watermark'),
]

# Serve static and media files only during development (DEBUG = True)
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)