from django.urls import path
from .views import classify_text, evaluate_user_response

urlpatterns = [
    path('mejor_consulta/', classify_text, name='mejor_consulta'), 
    path('mejor_respuesta/', evaluate_user_response, name='mejor_respuesta'),  
]
