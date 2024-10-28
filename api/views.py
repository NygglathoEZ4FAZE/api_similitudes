from django.http import JsonResponse
import torch
import json
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from django.views.decorators.csrf import csrf_exempt
from .initialize import tokenizaer_loader, model_loader # Importar el model_loader

vectorizer = TfidfVectorizer()

# Funciones para traer la mejor respuesta en base a una consulta
def get_embeddings(text_list):
    embeddings = vectorizer.transform(text_list)  # Vectorizar el texto (ajusta según tu modelo)
    return torch.tensor(embeddings.toarray(), dtype=torch.float32)

def cosine_similarity(query_vector, result_vectors):
    similarities = F.cosine_similarity(query_vector, result_vectors, dim=-1)
    return similarities

def find_best_response(query, instrucciones, respuestas):
    # Verificar que haya instrucciones y respuestas disponibles
    if not instrucciones or not respuestas:
        return "No se encontraron resultados para el intent."
    
    vectorizer.fit(instrucciones)

    # Obtener embeddings de la consulta y de las instrucciones
    query_embedding = get_embeddings([query])  # Embedding de la consulta
    result_embeddings = get_embeddings(instrucciones)  # Embeddings de las instrucciones
    
    # Asegurarse de que las dimensiones coincidan
    query_embedding = query_embedding.squeeze()  # Asegurar que sea 1D
    result_embeddings = result_embeddings.squeeze(1)  # Asegurarse de que sea 2D

    # Calcular la similitud coseno
    similarities = cosine_similarity(query_embedding, result_embeddings)

    # Imprimir las similitudes para depuración
    print("Similitudes:", similarities)

    # Verificar que similarities tenga valores válidos
    if similarities.numel() == 0:
        return "Error: No se calcularon similitudes."

    # Encontrar el índice de la instrucción más similar
    best_match_index = similarities.argmax().item()
    
    # Comprobar si la similitud máxima es válida
    best_similarity = similarities[best_match_index].item()
    
    if best_similarity == 0:
        return "No se encontraron coincidencias satisfactorias."

    # Devolver la respuesta correspondiente a la instrucción más similar
    best_response = respuestas[best_match_index]
    
    return best_response


def get_embedding_response(text):
    if isinstance(text, str):
        inputs = tokenizaer_loader(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model_loader(**inputs)
        
        # Calcular la media de los logits como embedding
        embedding = outputs.logits.mean(dim=1)  # Usar logits como embeddings
        
        return embedding
    else:
        raise ValueError("La entrada debe ser una cadena de texto.")
    
def find_best_response_response(user_response, id, instrucciones, respuestas, threshold=0.8):
    if not isinstance(user_response, str):
        raise ValueError("user_response debe ser una cadena de texto.")
    
    response_vector = get_embedding_response(user_response)
    respuestas_vectors = []

    for resp in respuestas:
        if isinstance(resp, str):
            embedding = get_embedding_response(resp)
            respuestas_vectors.append(embedding)
        else:
            raise ValueError("Cada respuesta debe ser una cadena de texto.")

    respuestas_vectors = torch.vstack(respuestas_vectors)
    similarities = F.cosine_similarity(response_vector, respuestas_vectors)

    # Filtrar respuestas por el umbral y obtener las similitudes correspondientes
    valid_indices = [i for i, score in enumerate(similarities) if score >= threshold]
    valid_similarities = [similarities[i] for i in valid_indices]

    if not valid_indices:
        # Si no hay coincidencias válidas
        best_response = None
        best_id = None
        best_instruction = None
    else:
        # Obtener el índice del valor máximo en valid_similarities
        best_match_index_in_valid = torch.argmax(torch.tensor(valid_similarities)).item()
        best_match_index = valid_indices[best_match_index_in_valid]
        
        # Obtener la mejor respuesta, su id y su instrucción
        best_response = respuestas[best_match_index]
        best_id = id[best_match_index]
        best_instruction = instrucciones[best_match_index]

    return {
        'best_id': best_id,
        'best_instruction': best_instruction,
        'best_response': best_response
    }

#Clases para cada función
@csrf_exempt
def classify_text(request):
    if request.method == 'POST':
        try:
            # Procesar el JSON enviado en el cuerpo de la solicitud
            data = json.loads(request.body)
            query = data.get('query', '')
            instrucciones = data.get('instrucciones', [])
            respuestas = data.get('respuestas', [])
            
            # Aquí llamarías a tu función de clasificación, por ejemplo:
            best_response = find_best_response(query, instrucciones, respuestas)

            # Devolver una respuesta en formato JSON
            return JsonResponse({'best_response': best_response}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'JSON inválido'}, status=400)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            return JsonResponse({'error': 'Error en el servidor: ' + str(e)}, status=500)

    return JsonResponse({'error': 'Método no permitido. Usa POST.'}, status=405)


@csrf_exempt
def evaluate_user_response(request):
    if request.method == 'POST':
        try:
            # Obtener los datos del cuerpo de la solicitud
            data = json.loads(request.body)
            user_response = data.get("user_response")
            id = data.get("id")
            instrucciones = data.get("instrucciones")
            respuestas = data.get("respuestas")

            result = find_best_response_response(user_response, id, instrucciones, respuestas)
            return JsonResponse(result)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)