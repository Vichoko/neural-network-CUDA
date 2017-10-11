# Redes neuronales en GPU con CUDA
Autor: Vicente Oyanedel M.

Profesor: Alexandre Bergel.

Curso: Redes Neuronales y Programación Genética; Departamento de Ciencias de la Computación; Universidad de Chile.

Fecha: Octubre, 2017.

## Objetivo

Se implementa una red neuronal para procesarse de manera paralela en GPU, con el objetivo de comparar el aceleramiento ("speed up") que se puede alcanzar en el entrenamiento de estas; comparado con la versión secuencial en CPU.

La idea general, es que en el proceso de entrenamiento ocurren distintas fases que se pueden paralelizar de manera particular:

* Forward Feed: Se debe esperar a que la capa previa termine de alimentarse, antes de alimentar la siguiente. Se paraleliza sobre cada neurona de la capa; iterando por capas, dada la restricción.
* Back Propagation: Se debe esperar a que la capa sucesiva actualice su delta, antes de propagar el error. Se paraleliza sobre cada neurona de la capa; iterando por capas, dada la restricción.
* Actualizar pesos: Se paraleliza sobre todas las neuronas de la red; dado que solo depende de su información interna.

Y cada fase (de las 3 expuestas) debe terminar completamente, antes de continuar la siguiente.
Se implementa una versión paralela y una secuencial para cada una de las 3 fases; obteniendo métricas de tiempo que tarda cada una. Para luego comparar el desempeño, midiendo la "cantidad de neuronas evaluadas por segundo".

Se entrena sobre un data-set de mensajes de spam, dada la alta dimensionalidad del espacio de características (representación "bag of words" de 11000 dimensiones). Principalmente, porque esto último propone a utilizar una cantidad cercana de neuronas por capa, en la red. 
Esto es importante, dado que si la red neuronal posee pocas neuronas por capa, entonces es probable que el sobre costo de utilizar la GPU sea mayor que el aceleramiento que se alcance por utilizarla. 

En palabras simples, la forma de sacar provecho de la GPU es tener un tamaño de problema grande. En particular, como se paraleliza sobre las neuronas de cada capa; aumentar la cantidad de neuronas por capa aumenta el tamaño del problema. 

## Funcionalidad
**Que funcionó y que no**

*La implementación se hace en lenguaje CUDA para GPUs NVidia.*

**Se implementó y funciona:**

1. La red neuronal, manejando sus variables en arreglos y matrices; para cada neurona. 
2. El entrenamiento de la red.
	1. Métodos de forward feed, back propagation y actualizar pesos de manera paralela.
	2. Métodos de forward feed, back propagation y actualizar pesos de manera secuencial.
	3. Mecanismo para obtener métricas de desempeño.
4. Prueba con compuertas lógicas: OR y AND.
5. Preprocesamiento de los SMS de SPAM y HAM.
    1. Filtro de caracteres especiales.
    2. Filtro de stop-words.
    3. Stemming de Porter.
    4. Calculo de diccionario y bag-of-words para mensajes de SPAM y HAM.

Para el mejor funcionamiento en GPU, se aplanaron las matrices a arreglos uni-dimensionales. Por lo que, **los deltas, pesos, outputs previos y biases**; se implementan como un arreglo plano, que se mapea a una neurona y una capa. 

Por esta optimización y por dificultades de implementación se introdujo la necesidad de **unificar la cantidad de neuronas por capa a una constante** (i.e. todas las capas tienen misma cantidad de neuronas, para que el mapeo sea directo).

Otra restricción que introduce esto, es que la **dimensión de output queda definida por la cantidad de neuronas por capa**.

**No funciona:**
1. Entrenamiento con bag of words (lenguaje natural).

Lamentablemente, luego de implementar la red neuronal, su entrenamiento y el preprocesamiento de datos se procedió con las pruebas de entrenamiento con lenguaje natural. Fase en la cual se notó que la implementación de la red neuronal no logra ejecutarse correctamente para dimensión de input rondando 11,000. 

Esto es lamentable dado que el principal objetivo de probar la red neuronal con un tamaño de problema grande queda imposible de satisfacer. Hasta el momento de la entrega fue imposible arreglar la implementación para poder hacer las mediciones necesarias.

Esta experiencia se analiza en más profundidad en la sección de *conclusiones*.

Por otro lado, se pudo probar el correcto funcionamiento de la implementación paralela y secuencial aprendiendo compuertas lógicas. 

## Código fuente y data set

El data set utilizado corresponden a 5,574 mensajes de texto (SMS) etiquetados como SPAM o HAM, el cual se puede encontrar en Kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset

El data set de compuertas logicas fue generado en el archivo ```cuda.cu```.

El código fuente suficiente para ejecutarlo se encuentra en mi repositorio: https://github.com/Vichoko/neural-network-CUDA

Para ejecutar el código se requiere una GPU NVidia con el SDK de CUDA y el data set. Mayor descripción de como ejecutarlo, se puede encontrar en el archivo **../readme.md**.


## Resultados y análisis

Se experimentó entrenando la red neuronal para aprender las compueras logicas OR y AND. Para ello se diseñó una red neuronal de 2 capas y 2 neuronas por capa. 
Se obtuvieron las siguientes metricas:

* CPU: 11,656,736 neuronas evaluadas por segundo, en promedio.
* GPU: 106,478 neuronas evaluadas por segundo, en promedio.

Se evidencia que la implementación en CPU completa el entrenamiento x109.5 veces más rápido que en GPU.

Esto ocurre dado que el sobre costo de mover datos de GPU a host, y viceversa; es mucho mayor que el costo de iterar por las pocas neuronas en la versión secuencial.

Como se mencionó, no se pudo obtener resultados con respecto al entrenamiento con datos de alta dimensionalidad; en el contexto de lograr el objetivo de hacer clasificación sobre texto natural, para mensajes de SPAM y HAM, y comparar el rendimiento.

![Imagen de entrenamiento con 2 neuronas y 2 capas; estadisticas de uso en CPU y entrenamiento en GPU funcionando](https://i.imgur.com/q2wCtFh.png)

## Análisis y Conclusión

El problema de convertir la red neuronal secuencial a la versión paralela en GPU requiere de un previo diseño detallado y completo de la implementación en GPU; antes de comenzar a implementar. Esto dado que el objetivo de particionar el problema en sub-problemas paralelizables, sujetos a las restricciones (no menores) que impone ejecutar aplicaciones en GPUs NVidia con CUDA; es un proceso complejo.

La implementación paralela en GPU fue diseñada con ayuda de material del curso de Programación en GPU, y material de apoyo encontrado en internet con respecto a la paralelización de redes neuronales mediante matrices.

Sin embargo, varias particularidades del lenguaje CUDA no fueron previstas hasta que ya se estaba implementado, debiendo tomar decisiones de diseño en el camino. Esto implicó restricciones imprevistas (como la cantidad fija de neuronas por capa, o la dimensión fija del output restringida a la cantidad de neuronas por capa). 

Relacionado con esto, no se previó completamente las restricciones de la codificación en CUDA, debiendo lidiar con muchos problemas de manejo de memoria y mapeo en los cores de la GPU, que disminuyeron fuertemente la velocidad de codificación. Debiendo invertir mucho tiempo en depuración y recodificación.

Por estas dificultades, se justifica el hecho que al entrenar con datos de alta dimensionalidad la implementación termina imprevistamente por problemas de direccionamiento de memoria. 

Sin embargo, la experiencia de programar una red neuronal en GPU me hizo incorporar realmente la lógica detrás del funcionamiento de estas (redes neuronales sigmoideas con entrenamiento por back propagation) de una manera integral. 
También me enseñó la lección de invertir más tiempo en el diseño, antes de comenzar a codificar; para evitar hacer rediseño durante la marcha. En especial con paradigmas de programación que no se está habituado.

Otro aprendizaje destacable es la dificultad de manejo de memoria en arquitecturas con varias memorias, las cuales no se pueden acceder desde cualquier parte. En particular porque mi experiencia había sido siempre programar en CPU (con una memoria RAM). Sin embargo, al tener memoria en el Host y en GPU, el paradigma se modifica y hay que pensar las cosas de una manera distinta a lo habitual.










