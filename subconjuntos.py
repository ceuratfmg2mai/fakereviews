import polars as pl

class SubconjuntoEquitativo:
    """
    Clase para crear un nuevo dataset muestreando registros de forma que,
    dentro de cada 'columna_categoria', las diferentes 'columna_clasificacion' 
    tengan un número equitativo de muestras, basado en la minoritaria.
    Utiliza Polars.
    """

    def __init__(self, 
                 df_original: pl.DataFrame, 
                 columna_categoria: str,
                 columna_clasificacion: str,
                 semilla_aleatoria: int = 42):
        """
        Inicializa el creador de subconjuntos.

        Args:
            df_original (pl.DataFrame): El DataFrame de entrada de Polars.
            columna_categoria (str): El nombre de la columna principal de agrupación (ej: "categories").
            columna_clasificacion (str): El nombre de la columna a equilibrar dentro de cada categoría (ej: "classification").
            semilla_aleatoria (int, optional): Semilla para operaciones aleatorias para asegurar la reproducibilidad.
                                                Defaults to 42.
        """
        self.df_original = df_original
        self.columna_categoria = columna_categoria
        self.columna_clasificacion = columna_clasificacion
        self.semilla_aleatoria = semilla_aleatoria
        
        self._validar_columnas_entrada()
        self.lista_categorias_unicas = self._extraer_categorias_unicas()

    def _validar_columnas_entrada(self):
        """Valida si las columnas especificadas existen en el DataFrame."""
        if self.columna_categoria not in self.df_original.columns:
            raise ValueError(
                f"La columna de categoría '{self.columna_categoria}' no se encuentra en el DataFrame."
            )
        if self.columna_clasificacion not in self.df_original.columns:
            raise ValueError(
                f"La columna de clasificación '{self.columna_clasificacion}' no se encuentra en el DataFrame."
            )

    def _extraer_categorias_unicas(self) -> list:
        """Extrae y reporta las categorías únicas de la columna_categoria."""
        try:
            lista_categorias = self.df_original.get_column(
                self.columna_categoria
            ).unique(maintain_order=True).to_list()
        except pl.exceptions.ColumnNotFoundError as e: # Debería ser capturado por _validar_columnas_entrada
            raise ValueError(f"Error al acceder a la columna de categoría '{self.columna_categoria}': {e}")

        num_categorias = len(lista_categorias)
        if num_categorias == 0:
            print("Advertencia: No se encontraron categorías únicas en la columna "
                  f"'{self.columna_categoria}'. Se devolverá un DataFrame vacío.")
            return []

        preview_categorias = lista_categorias[:5]
        sufijo_preview = "..." if num_categorias > 5 else ""
        print(
            f"Se encontraron {num_categorias} categorías únicas en '{self.columna_categoria}'. "
            f"Ejemplos: {preview_categorias}{sufijo_preview}"
        )
        return lista_categorias

    def _determinar_n_por_clasificacion_en_categoria(self) -> dict:
        """
        Determina cuántas muestras tomar de cada 'columna_clasificacion' dentro de cada 'columna_categoria'.
        El número de muestras para cada clasificación dentro de una categoría se basa en el recuento
        de la clasificación minoritaria en esa categoría específica.

        Returns:
            dict: Un mapa donde las claves son tuplas (valor_categoria, valor_clasificacion) 
                  y los valores son el número de muestras a tomar.
        """
        mapa_n_objetivo = {}

        if not self.lista_categorias_unicas:
            return {}

        for cat_valor in self.lista_categorias_unicas:
            # Filtrar por la categoría actual
            df_filtrado_por_categoria: pl.DataFrame
            if cat_valor is None:
                df_filtrado_por_categoria = self.df_original.filter(pl.col(self.columna_categoria).is_null())
            else:
                df_filtrado_por_categoria = self.df_original.filter(pl.col(self.columna_categoria) == cat_valor)

            if df_filtrado_por_categoria.height == 0:
                # print(f"Info: Categoría '{cat_valor}' está vacía en el DataFrame original. No se muestreará.")
                continue
            
            # Obtener cuentas de la columna_clasificacion dentro de esta categoría
            cuentas_clasificacion_en_categoria = df_filtrado_por_categoria.get_column(
                self.columna_clasificacion
            ).value_counts()

            if cuentas_clasificacion_en_categoria.height == 0:
                print(f"Advertencia: Categoría '{cat_valor}' no tiene valores de clasificación en "
                      f"'{self.columna_clasificacion}'. No se muestreará nada de esta categoría.")
                continue
            
            # Determinar el número mínimo de muestras (basado en la clasificación minoritaria)
            n_a_tomar_para_esta_categoria = cuentas_clasificacion_en_categoria.get_column("count").min()
            
            if n_a_tomar_para_esta_categoria == 0: # Si alguna clasificación tiene 0, no se puede equilibrar.
                 print(f"Info: Para categoría '{cat_valor}', una clasificación tiene 0 instancias. "
                       f"Por lo tanto, se tomarán 0 muestras para mantener el equilibrio.")
            else:
                 print(f"Info: Para categoría '{cat_valor}', se tomarán {n_a_tomar_para_esta_categoria} muestras "
                       "de cada clasificación presente para equilibrar.")

            # Asignar este número de muestras a cada clasificación presente en la categoría
            for fila_cuenta in cuentas_clasificacion_en_categoria.iter_rows(named=True):
                clas_valor = fila_cuenta[self.columna_clasificacion]
                mapa_n_objetivo[(cat_valor, clas_valor)] = n_a_tomar_para_esta_categoria
        
        return mapa_n_objetivo

    def _muestrear_grupos_individuales(self, mapa_n_objetivo: dict) -> list[pl.DataFrame]:
        """
        Muestrea registros de cada grupo (categoria, clasificacion) según el mapa_n_objetivo.
        """
        lista_dataframes_muestreados = []

        if not mapa_n_objetivo:
             print("Advertencia: El mapa de objetivos de muestreo está vacío.")
             return []

        for (cat_valor, clas_valor), num_a_muestrear in mapa_n_objetivo.items():
            if num_a_muestrear == 0:
                continue # No muestrear si el objetivo es 0 para esta combinación
            
            # Construir condiciones de filtrado manejando Nones
            condicion_cat: pl.Expr
            if cat_valor is None:
                condicion_cat = pl.col(self.columna_categoria).is_null()
            else:
                condicion_cat = pl.col(self.columna_categoria) == cat_valor
            
            condicion_clas: pl.Expr
            if clas_valor is None:
                condicion_clas = pl.col(self.columna_clasificacion).is_null()
            else:
                condicion_clas = pl.col(self.columna_clasificacion) == clas_valor
                
            df_grupo_especifico = self.df_original.filter(condicion_cat & condicion_clas)
            altura_grupo_actual = df_grupo_especifico.height

            if altura_grupo_actual == 0:
                # Esto no debería ocurrir si num_a_muestrear > 0 y _determinar_n... es correcto
                print(f"Advertencia Inesperada: El grupo ({cat_valor}, {clas_valor}) está vacío "
                      f"pero se solicitaron {num_a_muestrear} muestras.")
                continue
            
            # La altura_grupo_actual debe ser >= num_a_muestrear porque num_a_muestrear
            # es el mínimo de las cuentas de clasificación DENTRO de esa categoría.
            # Si esta condición se rompe, hay un error lógico en _determinar_n...
            if altura_grupo_actual < num_a_muestrear:
                 print(f"Advertencia de Lógica: Para ({cat_valor}, {clas_valor}), se necesitan {num_a_muestrear} "
                       f"pero solo hay {altura_grupo_actual}. Se tomarán {altura_grupo_actual}.")
                 # Esto implicaría que num_a_muestrear no fue realmente el mínimo para esta clas_valor.
                 # Se tomará lo disponible.
                 df_muestreado_grupo = df_grupo_especifico.sample(n=altura_grupo_actual, shuffle=True, seed=self.semilla_aleatoria)
            else:
                 df_muestreado_grupo = df_grupo_especifico.sample(n=num_a_muestrear, shuffle=True, seed=self.semilla_aleatoria)
            
            lista_dataframes_muestreados.append(df_muestreado_grupo)
            
        return lista_dataframes_muestreados

    def _ensamblar_subconjunto_final(self, 
                                   lista_dataframes_muestreados: list[pl.DataFrame]) -> pl.DataFrame:
        """
        Concatena los DataFrames de los grupos muestreados y mezcla el resultado final.
        """
        if not lista_dataframes_muestreados:
            print(
                "Advertencia: No se muestrearon datos (posiblemente todas las categorías/clasificaciones "
                "resultaron en 0 muestras objetivo). Devolviendo DataFrame vacío."
            )
            return pl.DataFrame() # Devuelve un DataFrame de Polars vacío

        df_final = pl.concat(lista_dataframes_muestreados)
        
        if df_final.height > 0: # Solo mezclar si hay datos
            df_final = df_final.sample(fraction=1.0, shuffle=True, seed=self.semilla_aleatoria) 
        
        print(f"El dataset final equilibrado contiene {df_final.height} registros.")
        return df_final

    def crear_subconjunto(self) -> pl.DataFrame:
        """
        Crea un nuevo dataset muestreando registros de forma equilibrada.
        Dentro de cada 'columna_categoria', las 'columna_clasificacion' se muestrean
        para tener el mismo número de instancias, basado en la minoritaria.

        Returns:
            pl.DataFrame: Un nuevo DataFrame de Polars con registros muestreados.
                          Devuelve un DataFrame vacío si no se pueden generar muestras.
        """
        if not self.lista_categorias_unicas: # Verificado en _extraer_categorias_unicas
            return pl.DataFrame()

        mapa_n_objetivo = self._determinar_n_por_clasificacion_en_categoria()
        
        if not mapa_n_objetivo:
             print("Advertencia: No se pudieron determinar objetivos de muestreo. "
                   "Devolviendo un DataFrame vacío.")
             return pl.DataFrame()

        lista_grupos_muestreados = self._muestrear_grupos_individuales(mapa_n_objetivo)
        
        df_subconjunto = self._ensamblar_subconjunto_final(lista_grupos_muestreados)
        
        return df_subconjunto


class SubconjuntoEquitativoSimple:
    """
    clase para crear un nuevo dataset muestreando registros de forma equitativa 
    de cada categoría utilizando.
    """

    def __init__(self, 
                 df_original: pl.DataFrame, 
                 columna_categoria: str, 
                 semilla_aleatoria: int = 32):
        """
        Inicializa el creador de subconjuntos.

        Args:
            df_original (pl.DataFrame): El DataFrame de entrada de Polars.
            columna_categoria (str): El nombre de la columna que contiene las etiquetas de categoría.
            semilla_aleatoria (int, optional): Semilla para operaciones aleatorias para asegurar la reproducibilidad.
                                                Defaults to 32.
        """
        self.df_original = df_original
        self.columna_categoria = columna_categoria
        self.semilla_aleatoria = semilla_aleatoria
        
        self._validar_columna_categoria()
        self.lista_categorias_unicas, self.num_categorias_unicas = self._extraer_y_validar_categorias()

    def _validar_columna_categoria(self):
        """Valida si la columna de categoría especificada existe en el DataFrame."""
        if self.columna_categoria not in self.df_original.columns:
            raise ValueError(
                f"La columna de categoría '{self.columna_categoria}' no se encuentra en el DataFrame."
            )

    def _extraer_y_validar_categorias(self) -> tuple[list, int]:
        """
        Extrae las categorías únicas de la columna especificada, las valida
        e informa al usuario sobre las categorías encontradas.
        """
        try:
            categorias_unicas_series = self.df_original.get_column(
                self.columna_categoria
            ).unique(maintain_order=True)
        except pl.exceptions.ColumnNotFoundError:
            # Esta excepción debería ser capturada por _validar_columna_categoria,
            # pero se mantiene por robustez.
            raise ValueError(
                f"Error al acceder a la columna de categoría '{self.columna_categoria}'."
            )
        
        lista_categorias_unicas = categorias_unicas_series.to_list()
        num_categorias_unicas = len(lista_categorias_unicas)

        if num_categorias_unicas == 0:
            print(
                "Advertencia: No se encontraron categorías en la columna especificada. "
                "Se procederá, pero es probable que el resultado sea un DataFrame vacío."
            )
            return [], 0

        preview_categorias = lista_categorias_unicas[:5]
        sufijo_preview = "..." if num_categorias_unicas > 5 else ""
        print(
            f"Se encontraron {num_categorias_unicas} categorías únicas. Ej: {preview_categorias}{sufijo_preview}"
        )
        
        # Control de categorías específico de la función original
        if num_categorias_unicas != 32:
            print(
                f"Info: El número de categorías únicas encontradas ({num_categorias_unicas}) "
                f"es diferente de las 32 esperadas, pero se procederá igualmente."
            )
        return lista_categorias_unicas, num_categorias_unicas

    def _calcular_mapa_muestreo(self, registros_totales_objetivo: int) -> dict:
        """
        Calcula cuántos registros se deben muestrear por cada categoría para alcanzar
        el número total de registros objetivo, distribuyendo los extras equitativamente.
        """
        if self.num_categorias_unicas == 0:
            return {}

        registros_base_por_categoria = registros_totales_objetivo // self.num_categorias_unicas
        num_categorias_con_registro_extra = registros_totales_objetivo % self.num_categorias_unicas

        mapa_registros_a_muestrear = {
            cat: registros_base_por_categoria for cat in self.lista_categorias_unicas
        }

        if num_categorias_con_registro_extra > 0:
            # Crear una Serie de Polars con las categorías únicas para usar su método .sample()
            serie_temporal_categorias_para_muestreo = pl.Series(
                "selector_categorias", self.lista_categorias_unicas
            )
            
            categorias_con_extra = serie_temporal_categorias_para_muestreo.sample(
                n=num_categorias_con_registro_extra,
                shuffle=True, # shuffle=True es el comportamiento por defecto cuando se especifica 'n'
                seed=self.semilla_aleatoria 
            ).to_list()
            
            for cat in categorias_con_extra:
                mapa_registros_a_muestrear[cat] += 1
        
        print(f"Registros objetivo: {registros_totales_objetivo}")
        if self.num_categorias_unicas > 0:
            print(f"Registros base por categoría: {registros_base_por_categoria}")
            print(
                f"{num_categorias_con_registro_extra} categorías aportarán "
                f"{registros_base_por_categoria + 1} registros."
            )
            print(
                f"{self.num_categorias_unicas - num_categorias_con_registro_extra} categorías aportarán "
                f"{registros_base_por_categoria} registros."
            )
            
        return mapa_registros_a_muestrear

    def _muestrear_grupos(self, mapa_registros_a_muestrear: dict) -> list[pl.DataFrame]:
        """
        Muestrea registros de cada categoría según el mapa de muestreo.
        Maneja casos donde los registros disponibles son menores a los solicitados.
        """
        lista_grupos_muestreados = []

        if not self.lista_categorias_unicas or not mapa_registros_a_muestrear:
             print("Advertencia: No hay categorías para muestrear o el mapa de muestreo está vacío.")
             return []

        for categoria_valor in self.lista_categorias_unicas:
            num_a_muestrear = mapa_registros_a_muestrear.get(categoria_valor, 0)
            
            if num_a_muestrear == 0:
                # Esto puede ocurrir si registros_totales_objetivo es menor que num_categorias_unicas
                print(f"Info: No se muestrearán registros para la categoría '{categoria_valor}' "
                      "según el plan de muestreo (0 registros asignados).")
                continue
            
            df_grupo = self.df_original.filter(pl.col(self.columna_categoria) == categoria_valor)
            altura_grupo_actual = df_grupo.height

            if altura_grupo_actual == 0:
                print(f"Advertencia: La categoría '{categoria_valor}' está vacía. No se pueden muestrear registros.")
                continue

            if altura_grupo_actual < num_a_muestrear:
                print(
                    f"Advertencia: La categoría '{categoria_valor}' tiene solo {altura_grupo_actual} registros, "
                    f"pero se solicitaron {num_a_muestrear}. Se tomarán todos los {altura_grupo_actual} disponibles."
                )
                if altura_grupo_actual > 0: # Solo muestrear si hay registros
                     lista_grupos_muestreados.append(
                        df_grupo.sample(n=altura_grupo_actual, shuffle=True, seed=self.semilla_aleatoria)
                    )
            else:
                lista_grupos_muestreados.append(
                    df_grupo.sample(n=num_a_muestrear, shuffle=True, seed=self.semilla_aleatoria)
                )
        return lista_grupos_muestreados

    def _ensamblar_subconjunto_final(self, 
                                   lista_grupos_muestreados: list[pl.DataFrame], 
                                   registros_totales_objetivo: int) -> pl.DataFrame:
        """
        Concatena los DataFrames de los grupos muestreados, mezcla el resultado final
        y emite una advertencia si el tamaño no coincide con el objetivo.
        """
        if not lista_grupos_muestreados:
            print(
                "Advertencia: No se muestrearon datos (posiblemente todas las categorías estaban vacías "
                "o no se solicitaron suficientes registros). Devolviendo DataFrame vacío."
            )
            return pl.DataFrame() # Devuelve un DataFrame de Polars vacío

        df_final = pl.concat(lista_grupos_muestreados)
        
        # Mezclar aleatoriamente el dataset final completo
        df_final = df_final.sample(fraction=1.0, shuffle=True, seed=self.semilla_aleatoria) 
        
        registros_reales = df_final.height
        if registros_reales != registros_totales_objetivo:
            print(
                f"Advertencia: El dataset final contiene {registros_reales} registros. "
                f"El objetivo era {registros_totales_objetivo}. "
                f"La diferencia puede deberse a registros insuficientes en una o más categorías."
            )
        return df_final

    def crear_subconjunto(self, registros_totales_objetivo: int) -> pl.DataFrame:
        """
        Crea un nuevo dataset muestreando registros de forma equitativa de cada categoría.

        Args:
            registros_totales_objetivo (int): El número total deseado de registros 
                                              en el nuevo dataset.

        Returns:
            pl.DataFrame: Un nuevo DataFrame de Polars con registros muestreados equitativamente.
                          Devuelve un DataFrame vacío si no se pueden generar muestras
                          (p.ej., no hay categorías, objetivo de 0 registros, etc.).
        """
        if registros_totales_objetivo <= 0:
            print("Advertencia: El número de registros totales objetivo es 0 o negativo. "
                  "Devolviendo un DataFrame vacío.")
            return pl.DataFrame()

        if self.num_categorias_unicas == 0:
             print("Advertencia: No hay categorías únicas para muestrear. "
                   "Devolviendo un DataFrame vacío.")
             return pl.DataFrame()

        mapa_registros_a_muestrear = self._calcular_mapa_muestreo(registros_totales_objetivo)
        
        # Si el mapa está vacío pero se esperaban categorías y registros,
        # podría ser porque registros_totales_objetivo < num_categorias_unicas,
        # y algunas categorías obtuvieron 0. _calcular_mapa_muestreo ya informa esto.
        # _muestrear_grupos manejará correctamente las categorías con 0 muestras.

        lista_grupos_muestreados = self._muestrear_grupos(mapa_registros_a_muestrear)
        
        df_subconjunto = self._ensamblar_subconjunto_final(
            lista_grupos_muestreados, registros_totales_objetivo
        )
        
        return df_subconjunto
