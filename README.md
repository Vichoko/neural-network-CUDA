# neural-network-CUDA

Una red neuronal implementada en CUDA y C; con el objetivo de comparar el "speed up" que se puede alcanzar utilizando la GPU para entrenar en paralelo.

Para información detallada, le invito a leer el [**Reporte**](https://github.com/Vichoko/neural-network-CUDA/blob/master/reporte.md) de este proyecto.

## Generalidades
Fue implementado y probado en **Ubuntu 16.06** con **CUDA 8.0**; ejecutado en una GPU NVidia Geforce GTX 960m.


## Dependencias
* Tener tarjeta de video (GPU) NVidia.
* Tener instalado CUDA Toolkit 8.0.
* Tener compilador nvcc (/usr/local/cuda-8.0/bin/nvcc).

### Configurar makefile
Si hay problemas de compilación puede necesitarse configurar la ubicación del compilador **nvcc** en el makefile.

## Compilación
```
$ make
```

Limpiar con:
```
$ make clean
```

## Ejecución
Ejecutar un entrenamiento de la compuerta logica OR secuencialmente y luego en GPU. 
```
$ ./main
```
Se muestran estadísticas de la cantidad de neuronas evaluadas por segundo al terminar cada entrenamiento o al precionar ```CTRL+C```.


