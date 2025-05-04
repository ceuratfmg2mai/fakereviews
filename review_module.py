import pandas as pd
from typing import Dict, List, Any, Set, Tuple


# # --- Cargar datos desde el archivo JSON ---
# file_path = 'base_reviews.json'
# review_data = {} # Inicializar como diccionario vacío

# try:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         # Cargar el contenido del archivo JSON en la variable review_data
#         review_data = json.load(f)
#     print(f"Datos cargados exitosamente desde {file_path}")
# except FileNotFoundError:
#     print(f"Error: El archivo '{file_path}' no fue encontrado.")
# except json.JSONDecodeError:
#     print(f"Error: El archivo '{file_path}' no contiene un JSON válido.")
# except Exception as e:
#     print(f"Ocurrió un error inesperado al cargar el archivo: {e}")
    

class ReviewFinder:
    """
    Clase para gestionar y buscar reseñas de negocios almacenadas
    en un DataFrame de pandas a partir de datos JSON estructurados.
    """
    def __init__(self, review_data: Dict[str, Dict[str, List[Dict[str, Any]]]]):
        """
        Inicializa la clase ReviewFinder.

        Args:
            review_data: Un diccionario que contiene los datos de las reseñas,
                         estructurado por categoría, y luego por 'fake_reviews'
                         y 'genuine_reviews'. Puede ser un diccionario vacío
                         si la carga del archivo falló.
        """
        if not review_data:
             print("Advertencia: Inicializando ReviewFinder con datos vacíos.")
             self.original_categories: set[str] = set() # Guardar categorías originales (vacío)
             self.df = pd.DataFrame()
        else:
            # Guardar las claves originales (nombres de categoría) como un conjunto para búsqueda eficiente
            self.original_categories: set[str] = set(review_data.keys())
            # Procesar los datos en el DataFrame
            self.df = self._preprocess_data(review_data)
        print(f"ReviewFinder inicializado con {len(self.original_categories)} categorías originales.")



    def _preprocess_data(self, data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> pd.DataFrame:
        """
        Convierte el diccionario de datos anidado en un DataFrame de pandas plano.

        Args:
            data: El diccionario de datos de reseñas original.

        Returns:
            Un DataFrame de pandas donde cada fila representa una reseña única.
        """
        all_reviews_list = []
        for category_name, reviews_dict in data.items():
            # Procesar reseñas falsas
            for review in reviews_dict.get('fake_reviews', []):
                review_copy = review.copy()
                review_copy['review_type'] = 'fake'
                if 'categories' not in review_copy:
                     review_copy['categories'] = [c.strip() for c in category_name.split(',')]
                all_reviews_list.append(review_copy)

            # Procesar reseñas genuinas
            for review in reviews_dict.get('genuine_reviews', []):
                review_copy = review.copy()
                review_copy['review_type'] = 'genuine'
                if 'categories' not in review_copy:
                     review_copy['categories'] = [c.strip() for c in category_name.split(',')]
                all_reviews_list.append(review_copy)

        return pd.DataFrame(all_reviews_list)

    def get_reviews_by_category(self, category_query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Busca y retorna reseñas falsas y genuinas que coincidan con una
        categoría o palabra clave de categoría.

        La búsqueda es sensible a mayúsculas/minúsculas y busca si la
        `category_query` está contenida en alguna de las categorías
        listadas para cada reseña.

        Args:
            category_query: La cadena de texto de la categoría a buscar.

        Returns:
            Un diccionario con dos claves: 'fake' y 'genuine'. Cada clave
            contiene una lista de diccionarios, donde cada diccionario
            representa una reseña que coincide con la consulta.
            Retorna listas vacías si no se encuentran coincidencias.
        """
        if self.df.empty:
            return {'fake': [], 'genuine': []}

        # Función de filtro: verifica si category_query está en alguna de las categorías de la fila
        def category_filter(row):
            separator = ", "
            categories_list = row['categories']
            joined_category_string = separator.join(categories_list)
            if joined_category_string == category_query:
                return True
            return False 
        
        # Aplicar el filtro
        matching_reviews_df = self.df[self.df.apply(category_filter, axis=1)]
        # Separar en falsas y genuinas
        fake_reviews_df = matching_reviews_df[matching_reviews_df['review_type'] == 'fake']
        genuine_reviews_df = matching_reviews_df[matching_reviews_df['review_type'] == 'genuine']

        # Convertir los DataFrames filtrados a listas de diccionarios
        fake_list = fake_reviews_df.drop(columns=['review_type']).to_dict('records')
        genuine_list = genuine_reviews_df.drop(columns=['review_type']).to_dict('records')

        return {'fake': fake_list, 'genuine': genuine_list}
    

    def verify_categories_exist(self, categories_to_check: List[str]) -> Tuple[bool, Set[str]]:
        """
        Verifica si todas las categorías proporcionadas en la lista existen
        como claves principales en los datos JSON originales cargados.

        Args:
            categories_to_check: Una lista de nombres de categoría (string)
                                 para verificar.

        Returns:
            Una tupla:
            - El primer elemento es un booleano: True si TODAS las categorías
              de la lista existen, False si al menos una no existe.
            - El segundo elemento es un conjunto (set) que contiene los nombres
              de las categorías de la lista de entrada que NO se encontraron
              en los datos originales. Estará vacío si todas existen.
        """
        if not self.original_categories:
            print("Advertencia: No hay categorías originales cargadas para verificar.")
            # Si no se cargó nada, ninguna categoría 'existe' en el sentido de los datos cargados
            return (False, set(categories_to_check))

        check_set = set(categories_to_check)
        # Encontrar las categorías en check_set que NO están en self.original_categories
        missing_categories = check_set.difference(self.original_categories)
        # Alternativa: missing_categories = check_set - self.original_categories

        all_exist = not bool(missing_categories) # True si el conjunto de faltantes está vacío

        return (all_exist, missing_categories)