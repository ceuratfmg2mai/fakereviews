import polars as pl
from qdrant_client import QdrantClient, models
import uuid
import time
from typing import List, Dict, Any, Optional

class QdrantDataIngestor:
    def __init__(self, qdrant_url: str, collection_name: str, embedding_model: Any, qdrant_key: Optional[str] = None, vector_size: Optional[int] = None):
        """
        Inicializa el ingestor de Qdrant.

        Args:
            qdrant_url (str): URL del servidor Qdrant (ej. "http://localhost:6333" o URL de la nube).
            collection_name (str): Nombre de la colección en Qdrant.
            embedding_model (Any): Instancia del modelo de embedding. Debe tener un método `embed_query(text: str) -> List[float]`.
            qdrant_key (Optional[str]): API key para Qdrant Cloud. Default None.
            vector_size (Optional[int]): Tamaño del vector. Si es None, se intentará inferir del modelo.
        """
        self.qdrant_url = qdrant_url
        self.qdrant_key = qdrant_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_key,
            timeout=60
        )
        
        if vector_size:
            self.vector_size = vector_size
        else:
            try:
                # Intenta inferir el tamaño del vector del modelo
                test_vector = self.embedding_model.embed_query("test text")
                self.vector_size = len(test_vector)
                print(f"Tamaño del vector inferido del modelo: {self.vector_size}")
            except Exception as e:
                raise ValueError(f"No se pudo inferir vector_size del embedding_model y no se proporcionó: {e}")

        self.points_batch_errors = [] # Para almacenar lotes que fallaron

    def setup_collection(self, payload_fields_to_index: Optional[Dict[str, models.PayloadSchemaType]] = None):
        """
        Crea la colección en Qdrant si no existe y configura los índices de payload.

        Args:
            payload_fields_to_index (Optional[Dict[str, models.PayloadSchemaType]]):
                Un diccionario donde las claves son nombres de campos del payload y los
                valores son los tipos de esquema de Qdrant (ej. models.PayloadSchemaType.KEYWORD).
        """
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print(f"La colección '{self.collection_name}' ya existe.")
        except Exception as e: # Asumimos que si falla es porque no existe o hay error de conexión
            if "not found" in str(e).lower() or "404" in str(e) or "status_code=404" in str(e):
                print(f"Creando colección '{self.collection_name}' con tamaño de vector {self.vector_size}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
                print(f"Colección '{self.collection_name}' creada.")
            else: # Otro tipo de error 
                print(f"Error al verificar la colección '{self.collection_name}': {e}")
                raise

        if payload_fields_to_index:
            print("Configurando índices de payload...")
            for field, schema_type in payload_fields_to_index.items():
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema_type
                    )
                    print(f"Índice de payload creado/verificado para el campo: {field}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"Índice para el campo '{field}' ya existía.")
                    else:
                        print(f"Advertencia: Error creando índice para '{field}': {e}")
        print("Configuración de la colección completada.")

    def _process_row(self, row_dict: Dict[str, Any]) -> Optional[models.PointStruct]:
        """Procesa una fila del DataFrame para convertirla en un PointStruct."""
        try:
            text_to_embed = row_dict['text']
            
            # Procesamiento especial de las categorías
            # categories_str = row_dict.get("categories") # Usar .get() para manejo seguro de claves faltantes
            # processed_categories = []
            # if isinstance(categories_str, str):
            #     processed_categories = [cat.strip() for cat in categories_str.split(',')]
            # elif isinstance(categories_str, list):
            #     processed_categories = [str(cat).strip() for cat in categories_str]
        
            print('Procesando fila con review_id:', row_dict.get('review_id', 'DESCONOCIDO'))
            # El payload dinámicamente a partir de las columnas del DataFrame
            payload = {
                "review_id": row_dict['review_id'],
                "review": text_to_embed, # Texto original de la reseña
                "user_id": row_dict['user_id'],
                "business_id": row_dict['business_id'],
                "review_count_user": int(row_dict['review_count_user']),
                "average_stars_user": float(row_dict['average_stars_user']),
                "yelping_days": float(row_dict['yelping_days']),
                "useful": float(row_dict['useful']),
                "funny": float(row_dict['funny']),
                "cool": float(row_dict['cool']),
                "fans": float(row_dict['fans']),
                "Cluster": float(row_dict['Cluster'])
            }
            
            # Generar embedding
            vector = self.embedding_model.embed_query(text_to_embed) # Usar el método del modelo
            
            # Generar un nuevo UUID para Qdrant
            qdrant_point_id = str(uuid.uuid4())
            
            return models.PointStruct(
                id=qdrant_point_id,
                vector=vector,
                payload=payload
            )
        except Exception as e:
            print(f"Error procesando fila con review_id '{row_dict.get('review_id', 'DESCONOCIDO')}': {e}")
            return None

    def ingest_dataframe(self, df: pl.DataFrame, batch_size: int = 256):
        """
        Ingesta un DataFrame de Polars en la colección de Qdrant.

        Args:
            df (pl.DataFrame): DataFrame de Polars con los datos a ingestar.
            batch_size (int): Tamaño de los lotes para la ingesta.
        """
        total_rows = len(df)
        if total_rows == 0:
            print("DataFrame vacío, no hay nada que ingestar.")
            return

        points_batch = []
        self.points_batch_errors = [] # Reiniciar lista de errores para esta ingesta
        start_time_total = time.time()

        print(f"\nIniciando ingestión de {total_rows} filas en la colección '{self.collection_name}'...")

        for i, row_dict in enumerate(df.iter_rows(named=True)):
            
            point = self._process_row(row_dict)
            if point:
                points_batch.append(point)
            
            if len(points_batch) >= batch_size or (i == total_rows - 1 and points_batch):
                try:
                    self.client.upsert(collection_name=self.collection_name, points=points_batch, wait=True)
                    print(f"Lote {i // batch_size + 1}/{ (total_rows + batch_size -1) // batch_size } (filas aprox. {i - len(points_batch) + 2}-{i + 1}) enviado. {len(points_batch)} puntos.")
                except Exception as e:
                    print(f"Error al ingestar el lote de puntos (filas aprox. {i - len(points_batch) + 2}-{i + 1}): {e}")
                    self.points_batch_errors.append(list(points_batch)) # Guardar una copia del lote
                points_batch = []
            
            if (i + 1) % (batch_size * 5) == 0 or i == total_rows - 1:
                elapsed_time_total = time.time() - start_time_total
                print(f"Progreso: {i + 1}/{total_rows} filas procesadas. Tiempo total: {elapsed_time_total:.2f} seg.")

        end_time_total = time.time()
        print(f"\nIngestión completada para {total_rows} filas.")
        print(f"Tiempo total de ingestión: {(end_time_total - start_time_total):.2f} segundos.")
        if self.points_batch_errors:
            num_failed_points = sum(len(batch) for batch in self.points_batch_errors)
            print(f"ATENCIÓN: {len(self.points_batch_errors)} lotes ({num_failed_points} puntos en total) fallaron durante la ingesta.")
            print("Puedes acceder a los lotes fallidos a través de la instancia: ingestor.points_batch_errors")
        else:
            print("Todos los lotes fueron procesados sin errores reportados por el cliente.")


    def verify_ingestion(self, original_id_to_check: Optional[str] = None): 
        """
        Verifica la ingesta contando los puntos y, opcionalmente, recuperando un punto
        específico basado en el 'review_id' original (almacenado en el payload con la clave 'review_id').
        Si no se proporciona original_id_to_check, recupera un punto usando scroll.
        """
        print("\n--- Verificación de Datos Insertados ---")
        try:
            count_result = self.client.count(collection_name=self.collection_name)
            print(f"Número total de puntos en la colección '{self.collection_name}': {count_result.count}")
            
            if count_result.count == 0:
                print("La colección está vacía, no se pueden verificar puntos.")
                return

            point_to_display = None 

            if original_id_to_check:
                print(f"Intentando recuperar la reseña con 'review_id' (en payload) igual a: '{original_id_to_check}'")
                
                # Crear un filtro para buscar por 'review_id' en el payload
                filtro_por_id_payload = models.Filter(
                    must=[                                
                        models.FieldCondition(
                            key="review_id", # Esta es la CLAVE DENTRO DE TU PAYLOAD
                            match=models.MatchValue(value=str(original_id_to_check)) # Asegurar que el valor sea string
                        )
                    ]
                )
                
                # Realizar la consulta
                query_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query_filter=filtro_por_id_payload,
                    limit=1, # Debería haber solo uno si 'review_id' en el payload es único
                    with_payload=True
                )
                
                if query_response.points:
                    point_to_display = query_response.points[0] # query_response.points es una lista
                    print(f"Punto encontrado por 'review_id' en payload.")
                else:
                    print(f"No se encontró ninguna reseña con 'review_id' (en payload) igual a: '{original_id_to_check}'.")
            
            else: # Si no se proporcionó un ID específico, tomar uno con scroll para verificación general
                print("No se proporcionó un 'review_id' específico para verificar. Recuperando un punto de muestra usando scroll...")
                scroll_response = self.client.scroll(
                    collection_name=self.collection_name, 
                    limit=1, 
                    with_payload=True
                )
                if scroll_response.points: # Acceder a .points del objeto ScrollResponse
                    point_to_display = scroll_response.points[0]
                    print(f"Mostrando un punto de muestra obtenido por scroll (ID Qdrant: {point_to_display.id}).")
                else:
                    print("No se pudieron obtener puntos de la colección para mostrar como muestra.")

            if point_to_display:
                print("\n--- Detalles del Punto de Muestra ---")
                print(f"  ID de Qdrant (UUID generado): {point_to_display.id}")
                print(f"  Payload:")
                if point_to_display.payload:
                    for key, value in point_to_display.payload.items():
                        print(f"    {key}: {value}")
                else:
                    print("    Payload vacío.")
        
        except Exception as e:
            print(f"Error al verificar datos en Qdrant: {e}")
        
    def get_classification_by_review_id(self, review_id: str) -> Optional[List[str]]:
        """
        Busca una reseña por el campo 'review_id' en el payload y retorna el campo 'classification'.

        Args:
            review_id (str): El 'review_id' de la reseña a buscar.

        Returns:
            Optional[List[str]]: La lista de categorías asociadas a la reseña, o None si no se encuentra.
        """
        try:
            # Crear un filtro para buscar por 'review_id' en el payload
            filtro_por_id_payload = models.Filter(
                must=[
                    models.FieldCondition(
                        key="review_id",
                        match=models.MatchValue(value=str(review_id))
                    )
                ]
            )

            # Realizar la consulta
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=filtro_por_id_payload,
                limit=1,
                with_payload=True
            )

            if query_response.points:
                point = query_response.points[0]
                if point.payload and "Cluster" in point.payload:
                    return point.payload["Cluster"]
                else:
                    print(f"La reseña con 'review_id' '{review_id}' no tiene el campo 'Cluster' en el payload.")
                    return None
            else:
                print(f"No se encontró ninguna reseña con 'review_id' igual a: '{review_id}'.")
                return None

        except Exception as e:
            print(f"Error al buscar la reseña con 'review_id' '{review_id}': {e}")
            return None
        
    def delete_collection(self):
        """
        Elimina la colección de Qdrant.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Colección '{self.collection_name}' eliminada.")
        except Exception as e:
            print(f"Error al eliminar la colección '{self.collection_name}': {e}")

    def get_review_by_text(self, review_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Busca las 5 reseñas más similares por el campo 'text' usando búsqueda vectorial y retorna sus payloads.

        Args:
            review_text (str): El texto de la reseña a buscar.

        Returns:
            Optional[List[Dict[str, Any]]]: Lista de payloads de las reseñas encontradas, o None si no se encuentra ninguna.
        """
        try:
            vector_de_consulta = self.embedding_model.embed_query(review_text)
            
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector_de_consulta,
                limit=5,
                with_payload=True
            )

            if query_response.points:
                return [point.payload for point in query_response.points if point.payload]
            else:
                print(f"No se encontraron reseñas similares al texto especificado.")
                return None

        except Exception as e:
            print(f"Error al buscar reseñas por texto: {e}")
            return None
