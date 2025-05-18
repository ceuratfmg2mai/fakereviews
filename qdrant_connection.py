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
            else: # Otro tipo de error (ej. conexión, autenticación)
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
            categories_str = row_dict.get("categories") # Usar .get() para manejo seguro de claves faltantes
            processed_categories = []
            if isinstance(categories_str, str):
                processed_categories = [cat.strip() for cat in categories_str.split(',')]
            elif isinstance(categories_str, list):
                processed_categories = [str(cat).strip() for cat in categories_str]
            
            # El payload dinámicamente a partir de las columnas del DataFrame
            payload = {
                "review_id": row_dict['review_id'],
                "review": text_to_embed, # Texto original de la reseña
                "user_id": row_dict['user_id'],
                "business_id": row_dict['business_id'],
                "stars": int(row_dict['stars']),
                "categories": processed_categories,
                "review_count_business": int(row_dict['review_count_business']),
                "stars_business": float(row_dict['stars_business']),
                "review_count_user": int(row_dict['review_count_user']),
                "average_stars_user": float(row_dict['average_stars_user']),
                "classification": row_dict['classification']
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
            # start_time_row = time.time() # Descomentar para timing por fila
            
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


    def verify_ingestion(self, sample_id_to_retrieve: Optional[str] = None):
        """
        Verifica la ingesta contando los puntos y recuperando un punto de muestra.
        Si se proporciona sample_id_to_retrieve, intenta recuperar ese ID específico de Qdrant.
        De lo contrario, recupera un punto aleatorio usando scroll.
        """
        print("\n--- Verificación de Datos Insertados ---")
        try:
            count_result = self.client.count(collection_name=self.collection_name)
            print(f"Número total de puntos en la colección '{self.collection_name}': {count_result.count}")
            
            if count_result.count > 0:
                point_to_show = None
                if sample_id_to_retrieve:
                    print(f"Intentando recuperar el punto con ID Qdrant: {sample_id_to_retrieve}")
                    retrieved_points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[sample_id_to_retrieve],
                        with_payload=True
                    )
                    if retrieved_points:
                        point_to_show = retrieved_points[0]
                    else:
                        print(f"No se pudo recuperar el punto con ID Qdrant: {sample_id_to_retrieve}")
                
                if not point_to_show: # Si no se especificó ID o no se encontró, tomar uno con scroll
                    print("Recuperando un punto de muestra usando scroll...")
                    scroll_response = self.client.scroll(
                        collection_name=self.collection_name, 
                        limit=1, 
                        with_payload=True
                    )
                    if scroll_response.points: # Acceder a .points
                        point_to_show = scroll_response.points[0]
                    else:
                        print("No se pudieron obtener puntos de la colección para mostrar.")

                if point_to_show:
                    print("\nEjemplo de punto recuperado:")
                    print(f"  ID de Qdrant: {point_to_show.id}")
                    print(f"  Payload: {point_to_show.payload}")
                else:
                    print("No se pudo recuperar un punto de muestra.")
            else:
                print("La colección está vacía, no se pueden verificar puntos.")
        except Exception as e:
            print(f"Error al verificar datos en Qdrant: {e}")