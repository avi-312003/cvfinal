from django.urls import path
from .views import home, upload_image, live_camera, upload_video

urlpatterns = [
    path('', home, name='home'),
    path('upload_image/', upload_image, name='upload_image'),
    path('live_camera/', live_camera, name='live_camera'),
    path('upload_video/', upload_video, name='upload_video'),
    
]
