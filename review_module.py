import polars as pl
from typing import Dict, Any

class ReviewFinder:
    """
    Clase para seleccionar aleatoriamente una reseña de un DataFrame de Polars
    y retornar un prompt cumplimentado con sus datos.
    """

    def __init__(self, df: pl.DataFrame):
        """
        Inicializa la clase ReviewFinder.

        Args:
            df: DataFrame de Polars con las reseñas.
        """
        if df.is_empty():
            print("Advertencia: Inicializando ReviewFinder con DataFrame vacío.")
        self.df = df

    def get_random_review_prompt(self, prompt_template: str) -> str:
        """
        Selecciona aleatoriamente una reseña usando semilla 42 y retorna el prompt cumplimentado.

        Args:
            prompt_template (str): Plantilla del prompt con llaves para los campos.

        Returns:
            str: Prompt cumplimentado con los datos de la reseña seleccionada.
        """
        if self.df.is_empty():
            return "No hay reseñas disponibles."

        # Selección aleatoria con semilla 42
        random_row = self.df.sample(n=1, seed=42).to_dicts()[0]

        # Cumplimentar el prompt
        prompt = prompt_template.format(
            text=random_row.get("text", ""),
            review_count_user=random_row.get("review_count_user", ""),
            average_stars_user=random_row.get("average_stars_user", ""),
            yelping_days=random_row.get("yelping_days", ""),
            useful=random_row.get("useful", ""),
            funny=random_row.get("funny", ""),
            cool=random_row.get("cool", ""),
            fans=random_row.get("fans", "")
        )
        return prompt, random_row.get("review_id", "")