# **BackendCanguro**
Backend para el proyecto de grado con la Fundación Canguro, desarrollado con **FastAPI**. Este backend permite ejecutar dos scripts (`script1.py` y `script2.py`) a través de endpoints REST.

## **Requisitos previos**
Antes de ejecutar el proyecto, asegúrate de tener lo siguiente instalado en tu sistema:
- Python 3.8 o superior.
- `pip` (administrador de paquetes de Python).
- Git (opcional si necesitas clonar el repositorio).

### **¿Cómo instalar `pip`?**
Si no tienes `pip` instalado, sigue estos pasos:

1. Verifica si `pip` ya está disponible ejecutando:
   ```bash
   python -m pip --version
   ```
2. Si no está instalado usa el siguiente comando:
   ```bash
   python -m ensurepip --upgrade
   ```
## **Cómo ejecutar el backend**

### **1. Clonar el repositorio**
Si este proyecto está alojado en un repositorio remoto (por ejemplo, GitHub), clona el repositorio:

```bash
git clone <https://github.com/sanvarp/BackendCanguro.git>
cd <NOMBRE_DEL_REPOSITORIO>
```
### **2. Crear un entorno virtual**
Crea un entorno virtual para aislar las dependencias del proyecto:

```bash
python -m venv venv
```
## **3. Acceder y activar el entorno virtual**
```bash
cd venv/Scripts 
```
```bash
activate.bat
```
Volver a la raiz del proyecto
```bash
cd ..
```
```bash
cd ..
```
### **4. Instalar dependencias**
Con el entorno virtual activado, instala las dependencias del proyecto especificadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **5. Ejecutar el servidor**
Para iniciar el servidor FastAPI, ejecuta el siguiente comando desde el directorio donde se encuentra el archivo `main.py`:

```bash
uvicorn main:app --reload
```
