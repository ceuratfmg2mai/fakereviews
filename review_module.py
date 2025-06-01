from typing import Dict, Any

class ReviewFinder:
    """
    Clase para crear el prompt de las reseñas test.
    """

    def build_prompt_from_row(self, row: Dict[str, Any], prompt_template: str) -> str:
        """
        Construye un prompt a partir de una fila y una plantilla.

        Args:
            row (dict): Fila del DataFrame (como diccionario).
            prompt_template (str): Plantilla del prompt con llaves para los campos.

        Returns:
            str: Prompt cumplimentado con los datos de la fila.
        """
        prompt = prompt_template.format(
            text=row.get("text", ""),
            review_count_user=row.get("review_count_user", ""),
            average_stars_user=row.get("average_stars_user", ""),
            yelping_days=row.get("yelping_days", ""),
            useful=row.get("useful", ""),
            funny=row.get("funny", ""),
            cool=row.get("cool", ""),
            fans=row.get("fans", "")
        )
        return prompt