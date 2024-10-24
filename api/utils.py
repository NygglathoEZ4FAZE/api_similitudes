from transformers import AutoTokenizer, AutoModelForTokenClassification

# Cargar el modelo y el tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    model = AutoModelForTokenClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    return tokenizer, model

# Cargar el modelo al iniciar
tokenizer, model_loader = load_model()