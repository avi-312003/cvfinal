"""
URL configuration for ArucoMarkerDetection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# ArucoMarkerDetection/urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from marker_app.views import home  # Import the home view


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='root'),
    path('marker/', include('marker_app.urls')),
    path('', RedirectView.as_view(url='marker/home')),  # Redirect to your desired URL
]

# ... (the rest of your URL patterns)

# ArucoMarkerDetection/urls.py
from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
