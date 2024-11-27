import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    make_scorer,
    confusion_matrix
)
import os
import json


# Función para cargar y preprocesar datos
def load_or_preprocess_data(filename, cache_filename, target_variable):
    if os.path.exists(cache_filename):
        df_original = pd.read_feather(cache_filename)
    else:
        df_original = pd.read_excel(filename, engine='openpyxl')
        df_original.replace('#NULL!', np.nan, inplace=True)
        df_original.dropna(subset=["V7", target_variable], inplace=True)
        df_original.reset_index(drop=True, inplace=True)
        df_original.to_feather(cache_filename)
    return df_original


# Función para seleccionar intervalos de fecha
def selector_de_intervalos(fecha1, fecha2, dataframe):
    return dataframe[(dataframe["V7"] > fecha1) & (dataframe["V7"] < fecha2)]


# Función principal
def run_analysis(filename, cache_filename, target_variable, variables_to_use, fecha_inicio="2000-01-01", fecha_fin="2025-01-01"):
    # Cargar datos
    df_original = load_or_preprocess_data(filename, cache_filename, target_variable)
    df_original.dropna(subset=["V7", target_variable], inplace=True)
    df_original["V7"] = pd.to_datetime(df_original["V7"], format='%d-%b-%y', errors='coerce')
    datos = selector_de_intervalos(fecha_inicio, fecha_fin, df_original)

    # Asegurar que la variable objetivo esté incluida
    if target_variable not in variables_to_use:
        variables_to_use.append(target_variable)
    datos = datos[variables_to_use]

    # Verificar que la variable objetivo está presente
    if target_variable not in datos.columns:
        raise KeyError(f"La variable objetivo '{target_variable}' no está en los datos.")

    # Identificar variables numéricas y categóricas
    numeric_columns = datos.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = datos.select_dtypes(exclude=[np.number]).columns.tolist()

    # Considerar variables numéricas con pocos valores distintos como categóricas
    for col in numeric_columns.copy():
        if datos[col].nunique() <= 16:
            categorical_columns.append(col)
            numeric_columns.remove(col)
    # Verificar si ambas listas están vacías
    if not numeric_columns and not categorical_columns:
        return None  # Salir de la función si no hay nada que procesar

    # Separar características y variable objetivo
    X = datos.drop(target_variable, axis=1)
    y = datos[target_variable]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if target_variable in categorical_columns:
        categorical_columns.remove(target_variable)
    if target_variable in numeric_columns:
        numeric_columns.remove(target_variable)

    # Procesar variables categóricas
    if categorical_columns:
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].astype(str)
            unseen_mask = ~X_test[col].isin(le.classes_)
            X_test.loc[unseen_mask, col] = 'unknown'
            le.classes_ = np.append(le.classes_, 'unknown')
            X_test[col] = le.transform(X_test[col])
            le_dict[col] = le
    

    # Imputar datos numéricos
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="mean")
        X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = numeric_imputer.transform(X_test[numeric_columns])


    # Balancear datos
    train_data = pd.concat([X_train, y_train], axis=1)

    # Contar la cantidad de ocurrencias de cada clase en el conjunto de entrenamiento
    frecuencias = train_data[target_variable].value_counts()

    # Determinar la clase mayoritaria
    max_value = frecuencias.max()

    # Realizar el oversampling para cada clase
    df_list = []

    for clase in frecuencias.index:
        df_clase = train_data[train_data[target_variable] == clase]
        if len(df_clase) < max_value:
            df_clase_oversampled = resample(
                df_clase,
                replace=True,
                n_samples=max_value,
                random_state=42
            )
        else:
            df_clase_oversampled = df_clase
        df_list.append(df_clase_oversampled)

    # Concatenar todas las clases
    train_data_balanced = pd.concat(df_list)
    X_train_balanced = train_data_balanced.drop(target_variable, axis=1)
    y_train_balanced = train_data_balanced[target_variable]
    # Entrenar modelo
    # Configuración del modelo con parámetros fijos
    model = RandomForestClassifier(
        n_estimators=100,       # Número de árboles en el bosque
        criterion="gini",       # Criterio de división
        max_depth=10,           # Profundidad máxima del árbol
        min_samples_split=5,    # Número mínimo de muestras requeridas para dividir un nodo
        min_samples_leaf=2,     # Número mínimo de muestras requeridas en un nodo hoja
        random_state=42         # Semilla para reproducibilidad
    )

    # Entrenar el modelo
    model.fit(X_train_balanced, y_train_balanced)

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)


    # Métricas y resultados
    f1 = f1_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    importances = model.feature_importances_
    feature_importance = dict(zip(X_train.columns, importances))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Retornar resultados
    return {
        "F1_score": f1,
        "recall":recall,
        "precision":precision,
        "Total muestras": len(y_test),
        "Falsos_positivos": int(fp),
        "Falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        "verdaderos_negativos": int(tn),
        "importancia_atributo": feature_importance
    }


# Bloque principal
if __name__ == "__main__":

    # Definir los argumentos
    parser = argparse.ArgumentParser(description="Script para análisis por intervalos.")
    parser.add_argument("--target_variable", type=str, required=True, help="Variable objetivo para el análisis.")
    parser.add_argument("--variables_to_use", type=str, required=True, help="Lista de variables a utilizar, separadas por comas.")
    parser.add_argument("--fecha_inicio", type=str, required=True, help="Fecha de inicio del rango (YYYY-MM-DD).")
    parser.add_argument("--fecha_fin", type=str, required=True, help="Fecha de fin del rango (YYYY-MM-DD).")
    parser.add_argument("--anos_por_periodo", type=int, required=True, help="Duración de cada intervalo en años.")

    args = parser.parse_args()

    filename = "base_de_datos.xlsx"
    cache_filename = "base_de_datos_precargado.feather"

    # Convertir variables de entrada
    target_variable = args.target_variable
    variables_to_use = args.variables_to_use.split(",")
    fecha_inicio = args.fecha_inicio
    fecha_fin = args.fecha_fin
    anos_por_periodo = args.anos_por_periodo

    # Convertir fechas a objetos datetime
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin)

    resultados = []

    # Dividir el rango en intervalos
    while fecha_inicio_dt < fecha_fin_dt:
        fecha_fin_intervalo = fecha_inicio_dt + pd.DateOffset(years=anos_por_periodo)

        if fecha_fin_intervalo > fecha_fin_dt:
            fecha_fin_intervalo = fecha_fin_dt


        try:
            # Ejecutar análisis para el intervalo actual
            resultado = run_analysis(
                filename,
                cache_filename,
                target_variable,
                variables_to_use,
                fecha_inicio=fecha_inicio_dt.strftime("%Y-%m-%d"),
                fecha_fin=fecha_fin_intervalo.strftime("%Y-%m-%d")
            )
            # Agregar fechas al resultado
            resultado["Fecha_ini"] = fecha_inicio_dt.date().isoformat()
            resultado["Fecha_fin"] = fecha_fin_intervalo.date().isoformat()
            resultados.append(resultado)
        except Exception as e:
            print(f"Error procesando el intervalo {fecha_inicio_dt.date()} - {fecha_fin_intervalo.date()}: {e}")

        # Actualizar inicio del siguiente intervalo
        fecha_inicio_dt = fecha_fin_intervalo

    # Imprimir resultados finales
    print("JSON final de resultados:")
    print(json.dumps(resultados, indent=4))
