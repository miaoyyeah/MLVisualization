from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view
import os
import openai
from . import gpt

def home(request):
    return HttpResponse("Welcome to the Home Page!")

@api_view(['POST'])
def upload_link(request):
    link = request.data.get('link')
    
    if not link:
        return JsonResponse({'error': 'No link provided'}, status=400)
    
    try:
        # Call the function from gpt.py to process the link
        json_file = gpt.url_to_json(link)
    except Exception as e:
        # Handle any exceptions raised by url_to_json
        return JsonResponse({'error': f'Failed to process link: {str(e)}'}, status=500)

    print("Link received successfully", link)
    return JsonResponse({'message': 'Link received successfully', 'link': link, "json_file": json_file}, status=200)
