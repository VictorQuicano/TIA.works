# 🚀 Introducción a CUDA

¡Bienvenido! Este repositorio contiene ejercicios prácticos para comenzar a trabajar con CUDA, la plataforma de cómputo paralelo de NVIDIA.

---

## 🛠️ Compilación y ejecución

> 💡 **TIP**  
> Puedes compilar los archivos usando: `bash nvcc [nombre_del_archivo].cu -o [nombre_del_ejecutable]`
> Luego, ejecuta el programa con: `bash ./[nombre_del_ejecutable]`

---

## 📚 Ejercicios del laboratorio

A continuación se presentan los trabajos desarrollados:

### 1. 🔍 Información del dispositivo CUDA

Obtener y mostrar las propiedades del dispositivo donde se está ejecutando el código.

![Dispositivo CUDA](/docs/pract_01/Ejer_01.png)

---

### 2. ➕ Suma de vectores con CUDA

Implementación de la suma de vectores utilizando programación paralela con CUDA.

![Suma de vectores](/docs/pract_01/Ejer_02.png)

---

### 3. ✖️ Multiplicación de matrices con CUDA

Implementación de la multiplicación de matrices utilizando hilos CUDA.

![Multiplicación de matrices](/docs/pract_01/Ejer_03.png)

---

## ✅ Requisitos

- GPU NVIDIA compatible con CUDA
- CUDA Toolkit instalado
- `nvcc` disponible en tu entorno de desarrollo

---

## 📎 Notas adicionales

Si tienes problemas al ejecutar el código o tu GPU no es detectada, asegúrate de:

- Tener los drivers de NVIDIA actualizados
- Verificar que `nvidia-smi` funciona correctamente
- Ejecutar los programas en un entorno que tenga acceso a la GPU

---

¡Listo para acelerar tu código con CUDA! 💻⚡
