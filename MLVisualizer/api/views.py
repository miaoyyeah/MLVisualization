from django.http import JsonResponse
from rest_framework.decorators import api_view
import os
import openai
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the Home Page!")

@api_view(['POST'])
def upload_link(request):
    link = request.data.get('link')
    if not link:
        return JsonResponse({'error': 'No link provided'}, status=400)
    
    print("Link received successfully", link)
    return JsonResponse({'message': 'Link received successfully', 'link': link})