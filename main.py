from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import json


# Instanciar la aplicación FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen. Cambia "*" a un dominio específico si es necesario.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.).
    allow_headers=["*"],  # Permite todos los encabezados.
)


# Modelo para recibir parámetros del primer script (simple)
class Script1Params(BaseModel):
    target_variable: str

# Modelo para recibir parámetros del segundo script (con periodos)
class Script2Params(BaseModel):
    target_variable: str
    variables_to_use: list
    fecha_inicio: str
    fecha_fin: str
    anos_por_periodo: int


@app.post("/run-script1/")
def run_script1(params: Script1Params):
    try:
        # Ruta absoluta del script1
        script1_path = os.path.abspath("script1.py")

        # Ejecutar script1 y capturar la salida
        result = subprocess.run(
            ["python", script1_path, params.target_variable],
            capture_output=True,
            text=True
        )

        # Si el script falla
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=result.stderr.strip())

        # Intentar parsear la salida para encontrar el JSON final
        output = result.stdout.strip()
        try:
            # Buscar el JSON en la salida capturada
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("No se encontró JSON en la salida del script.")
            
            json_output = output[json_start:json_end]
            parsed_json = json.loads(json_output)
            return parsed_json
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Error al parsear el JSON de salida del script1.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-script2/")
def run_script2(params: Script2Params):
    try:
        # Ruta absoluta del script2
        script2_path = os.path.abspath("script2.py")

        # Construir argumentos para el script2
        args = [
            "python",
            script2_path,
            "--target_variable", params.target_variable,
            "--variables_to_use", ",".join(params.variables_to_use),
            "--fecha_inicio", params.fecha_inicio,
            "--fecha_fin", params.fecha_fin,
            "--anos_por_periodo", str(params.anos_por_periodo),
        ]

        # Ejecutar script2 y capturar la salida
        result = subprocess.run(
            args,
            capture_output=True,
            text=True
        )

        # Si el script falla
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=result.stderr.strip())

        # Intentar parsear la salida para encontrar el JSON final
        output = result.stdout.strip()
        try:
            # Buscar el JSON en la salida capturada
            json_start = output.find("[")
            json_end = output.rfind("]") + 1
            if json_start == -1 or json_end == -1:
                raise ValueError("No se encontró JSON en la salida del script.")
            
            json_output = output[json_start:json_end]
            parsed_json = json.loads(json_output)
            return parsed_json
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Error al parsear el JSON de salida del script2.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ruta raíz para probar el estado del backend
@app.get("/")
def read_root():
    return {"message": "El backend está funcionando correctamente"}