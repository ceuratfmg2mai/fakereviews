from typing import Dict, Any
from qdrant_connection import QdrantDataIngestor
class ReviewFinder:
    """
    Clase para crear el prompt de las reseñas test.
    """

    def build_test_prompt_from(self, row: Dict[str, Any], prompt_template: str) -> str:
        """
        Construye un prompt a partir de una fila y una plantilla.

        Args:
            row (dict): Fila del DataFrame (como diccionario).
            prompt_template (str): Plantilla del prompt con llaves para los campos.

        Returns:
            str: Prompt cumplimentado con los datos de la fila.
        """
        prompt = prompt_template.format(
            text=row.get("text", "").replace('\n', ' '),
            review_count_user=row.get("review_count_user", ""),
            average_stars_user=row.get("average_stars_user", ""),
            yelping_days=row.get("yelping_days", ""),
            useful=row.get("useful", ""),
            funny=row.get("funny", ""),
            cool=row.get("cool", ""),
            fans=row.get("fans", "")
        )
        return prompt
    
    def format_principal_prompt_for(row_data: dict, ingestor:QdrantDataIngestor, prompt_template_example: str, base_prompt_template: str) -> str:
        """
        Toma un diccionario (row_data) con los datos de la fila y rellena la plantilla.
        Maneja valores nulos convirtiéndolos a string (p.ej., 'None').
        """
        # Crea una copia o un nuevo diccionario para asegurar que todos los valores sean strings
        # y manejar los nulos de forma segura para .format()
        data_for_format = {}
        final_prompt = None
        separator = ", "
        for key, value in row_data.items():
            if key == 'text':
                if not value:
                    return f"ERROR: El texto de la reseña no puede ser vacio"
                else:
                    # Busqueda por similitud en el rag
                    reviews = ingestor.get_review_by_text(str(value).replace('\n', ' '))
                    if reviews:
                        # creación del prompt example
                        prompts = ingestor.build_prompts_from_reviews(reviews, prompt_template_example)
                        data_for_format['reviews_examples_with_metadata'] = separator.join(prompts)                        
                    else:
                        return f"ERROR: No se encontraron reseñas de ejemplo en el RAG para la reseña: {value[:50]}..."        
            else:
                data_for_format['review_test'] = value     

        try:
            print(data_for_format)
            return base_prompt_template.format(**data_for_format)
        except KeyError as e:
            # Error útil si falta un placeholder en los datos de la fila
            print(f"Error: Falta la clave {e} en los datos de la fila para formatear la plantilla.")
            return f"ERROR_FORMATTING_MISSING_KEY_{e}"
        except Exception as e:
            print(f"Error inesperado al formatear: {e}")
            return "ERROR_FORMATTING_UNEXPECTED"