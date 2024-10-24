from django.apps import AppConfig


class HomeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    def ready(self):
            # Importar el archivo de inicialización
            import api.initialize