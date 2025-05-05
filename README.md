
# DESARROLLO TÉCNICO DEL TRABAJO FIN DE MASTER - DETECCIÓN DE DEEPFAKES EN RESEÑAS DE PRODUCTOS CON TRANSFORMER

Una reseña falsa se define como cualquier valoración en línea que sea engañosa, fabricada o incentivada de manera que no represente una experiencia genuina de un cliente. Estas reseñas carecen de autenticidad, ya sea porque el revisor no existe, porque la experiencia descrita nunca ocurrió, o porque el revisor recibió algún tipo de compensación o incentivo por emitir una opinión particular sin que esto se revele.

Hoy en día para cualquier empresa, sin importar el tamaño del mismo, las reseñas impactan directamente en su evolución.  Para las empresas, estas opiniones representan una valiosa retroalimentación que puede impulsar mejoras en sus ofertas y fortalecer su reputación en línea. Una calificación positiva y un flujo constante de comentarios favorables pueden traducirse en una mayor confianza del consumidor y, en última instancia, en un incremento de las ventas.

La proliferación de contenido generado por los usuarios que incluye comentarios, reseñas y opiniones en redes sociales, ha alcanzado un volumen tal que resulta abrumador para los usuarios que buscan información relevante.

En diversos tipos de modelos ML se han aplicado distintas técnicas de detección de reseñas falsas (ej: análisis a nivel de oración, análisis a nivel de aspecto o característica). Estos modelos se entrenan utilizando conjuntos de datos etiquetados, donde las reseñas se clasifican previamente como genuinas o falsas.  Los resultados obtenidos en algunos estudios sugieren que los clasificadores automáticos pueden alcanzar una precisión casi perfecta en la detección de reseñas falsas, superando significativamente o igualando la capacidad de los evaluadores humanos.

Una de las dificultades centrales en el desarrollo del TFM reside en la obtención de un conjunto de datos adecuadamente categorizado. En consecuencia, se emplearán estrategias de Ciencia de Datos e Inteligencia Artificial con el fin de generar un dataset idóneo para este trabajo.

## Desarrollo técnico

El desarrollo técnico de este TFM incloyó varias estrategias de selección, exploración y evaluación de los datos, además de la selección y ejecución de varios modelos ML. El desarrollo técnico está implementado de la siguiente forma:

[1] Procesamiento de datos: En este primer paso se llevó a cabo la carga del extenso DataSet Yelp y posteriormente la selección estratégica de un segmento de datos con el objetivo de facilitar la exploración, reducir los tiempos de procesamiento y enfocar el análisis en un subconjunto manejable y posiblemente relevante para los fines de este TFM.

[2] Categorización por DeepSeek del segmento de reseñas seleccionado: Partiendo de un segmento de 17.000 reseñas que seleccionamos (distribuidas en 32 tipos de establecimiento, ej: Spas, Restaurantes), DeepSeek se encargó de clasificarlas como falsas o genuinas. Para guiar a DeepSeek en esta tarea, preparamos un prompt específico aplicando la técnica 'few-shot prompting', el cual contenía ejemplos de ambos tipos de reseñas previamente seleccionados por tipo de establecimiento. 

## Referencias

[1] Analyzing sentiments in e-commerce: Techniques, applications and challenges, acceso: marzo 20, 2025, https://ijsra.net/sites/default/files/IJSRA-2024-0843.pdf

[2] Customer Review Analysis and Identifying Spam Reviews, acceso: marzo 20, 2025, https://ijarsct.co.in/A8908.pdf

[3] Creating and detecting fake reviews of online products, https://econpapers.repec.org/RePEc:eee:joreco:v:64:y:2022:i:c:s0969698921003374

[4] DataSet seleccionado de reseñas: https://www.kaggle.com/datasets/aagudelom/reviews-dataset

## Gracias:

[1] Seguros el Corte inglés, https://seguros.elcorteingles.es/

[2] CEURA. 

[3]  UAH, https://uah.es/es/

## License

[MIT](https://choosealicense.com/licenses/mit/)

