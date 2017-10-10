# neural-network-CUDA
A simple neural network written in CUDA and C.

## Generalidades
Implementado y probado en **Ubuntu 16.06** con **CUDA 1.8**.


## Dependencias
* Tener tarjeta de video NVidia.
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


