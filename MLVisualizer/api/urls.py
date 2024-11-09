from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload-link/', views.upload_link, name='upload_link'), 
]