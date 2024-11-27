import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import json


def run_analysis(target_variable):
    """
    Ejecuta el análisis completo utilizando la lógica original del script.
    :param target_variable: Nombre de la variable objetivo.
    :return: JSON con los resultados de importancia de características y correlaciones.
    """
    print("Iniciando análisis de datos...")
    # Verificar si se pasa la variable objetivo como argumento
    if not target_variable:
        raise ValueError("La variable objetivo no fue proporcionada.")

    # Función para cargar y preprocesar datos si no están en caché en formato Feather
    def load_or_preprocess_data(filename, cache_filename, target_variable):
        if os.path.exists(cache_filename):
            print("Cargando datos del archivo de caché Feather...")
            df_original = pd.read_feather(cache_filename)
        else:
            print("Cargando datos desde el archivo original y preprocesando...")
            df_original = pd.read_excel(filename, engine='openpyxl')
            df_original.replace('#NULL!', np.nan, inplace=True)
            df_original.dropna(subset=['V7', target_variable], inplace=True)
            print("Guardando datos procesados en formato Feather para uso futuro...")
            df_original.reset_index(drop=True, inplace=True)
            df_original.to_feather(cache_filename)
        return df_original

    # Cargar los datos
    filename = 'base_de_datos.xlsx'
    cache_filename = 'base_de_datos_precargado.feather'
    df_original = load_or_preprocess_data(filename, cache_filename, target_variable)
    print("Datos cargados correctamente.")

    # Crear una copia de trabajo
    df = df_original.copy()

    # Convertir 'V7' a formato de fecha
    print("Convirtiendo 'V7' a formato de fecha...")
    df['V7'] = pd.to_datetime(df['V7'], format='%d-%b-%y', errors='coerce')

    # Definir una función para seleccionar intervalos de nacimiento
    def selector_de_intervalos(fecha1, fecha2, dataframe):
        print(f"Seleccionando datos dentro del rango de fechas {fecha1} a {fecha2}...")
        dataframe = dataframe[(dataframe['V7'] > fecha1) & (dataframe['V7'] < fecha2)]
        return dataframe

    # Seleccionar datos dentro del rango de fechas especificado
    datos = selector_de_intervalos("2000-01-01", "2025-01-01", df)
    print("Selección de rango de fechas completa.")

    # Lista de columnas a eliminar
    variables_a_eliminar = [
        'Idenfinal', 'Iden_Codigo', 'Iden_Sede', 'V7', 'V7a', 'V259', 'V387', 'V227', 'V195A', 'V195',
        'HD_FechaEntrada', 'HD_FechaSalida', 'CD12', 'V390', 'V389', 'V391', 'ANOCAT2ISS', 'V195Bb',
        'V430C', 'V430B', 'examenneurodurante12meses', 'rsm12m', 'rsm6m', 'CD6', 'ANOCAT', 'vino12m',
        'examenneuropsico12meses', 'infanib9m', 'V429', 'V408', 'IQ6cat', 'infanib12m', 'V229',
        'infanib6m', 'V246', 'NEURO40', 'V230', 'V372', 'V430', 'V327', 'V410', 'V366', 'V292',
        'V412', 'V414', 'CSP_EscolaridadMadre', 'riesgoPC12m', 'v402', 'V344C', 'V413', 'resptometria',
        'V203', 'V236E', 'V388', 'V303', 'V389', 'V390', 'V326', 'V365', 'zscorefinalpesoparatalla12mesesOMS',
        '"zscoreBMI12mesessegunOMS', 'tallametro12m', 'BMI12meses', 'zscoreBMI12mesessegunOMS',
        'zscorepeso12mOMS', 'BMI_FOR_AGE_40semanas','zscorepesoparatalla9meses', 'pesokilo12m', 'zscoretalla12mOMS', 'CONSULT11',
        'zscorePerimetroC12mOMS','CONSULT10','CONSULT12','zscorepeso40SemOMS','RELACIONAROSOBRETOTALCONSULTAS','CONSULT09'
        'zscoretalla9mOMS','zscorepeso9mOMS','percapindicfinal','zscorefinalpesoparalatalla40semanas','Gananciatallaentradatalla40sem'
        ,'zscorePerimetroC9mOMS','IQ12cat','zscorepesoparatalla40semanas','V347']

    # Asegurar que la variable objetivo no se elimine
    if target_variable in variables_a_eliminar:
        print(f"Advertencia: La variable objetivo '{target_variable}' está en la lista de variables a eliminar. Se conservará.")
        variables_a_eliminar.remove(target_variable)

    # Eliminar columnas especificadas
    print("Eliminando columnas innecesarias...")
    datos = datos.drop(columns=variables_a_eliminar, errors='ignore')

    # Eliminar columnas con tipo de datos de fecha
    date_columns = datos.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    datos = datos.drop(columns=date_columns, errors='ignore')
    print("Columnas de tipo fecha eliminadas.")

    # Generar un subconjunto de datos para el cálculo de correlación, aplicando el filtro del 20%
    imputed_values = datos.isna().sum()
    total_rows = len(datos)
    imputed_percentage = (imputed_values / total_rows) * 100
    valid_columns = imputed_percentage[imputed_percentage < 20].index

    # Asegurarse de que la variable objetivo esté incluida en el conjunto de correlación
    if target_variable not in valid_columns:
        valid_columns = valid_columns.union([target_variable])

    datos_correlacion = datos[valid_columns]

    # Identificar columnas categóricas y numéricas para correlación
    print("Identificando columnas categóricas y numéricas...")

    categorical_columns = datos.select_dtypes(include=[np.number]).columns
    categorical_columns = [col for col in categorical_columns if datos[col].nunique() <= 16]
    categorical_columns = [col for col in categorical_columns if col != target_variable] 
    numeric_columns = [col for col in datos.select_dtypes(include=[np.number]).columns if col not in categorical_columns]
    numeric_columns = [col for col in numeric_columns if col != target_variable] 

    # Identificar columnas numéricas para el conjunto de correlación
    numeric_columns_corr = [col for col in datos_correlacion.select_dtypes(include=[np.number]).columns if col != target_variable]

    # Separar características y variable objetivo
    print("Separando características y variable objetivo...")
    X = datos.drop(target_variable, axis=1)
    y = datos[target_variable]

    # Calcular correlaciones y seleccionar las 20 variables principales por correlación con la variable objetivo
    correlations = datos_correlacion[numeric_columns_corr].corrwith(datos_correlacion[target_variable]).abs()
    top_20_correlated_vars = correlations.sort_values(ascending=False).head(20)

    # Imprimir las 20 variables con mayor correlación
    print("Top 20 variables por correlación con la variable objetivo:", top_20_correlated_vars.index.tolist())

    # Dividir el conjunto de datos en entrenamiento y prueba
    print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Codificando columnas categóricas con un enfoque optimizado...")

    # Codificar columnas categóricas en X_train y aplicar en X_test
    label_encoders = {}

    # Codificar cada columna categórica en X_train y aplicar en X_test sin usar apply()
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))  # Ajustar solo con X_train
        
        # Guardar el codificador en el diccionario para referencia futura
        label_encoders[col] = le

        # Transformar X_train
        X_train[col] = le.transform(X_train[col].astype(str))
        
        # Crear un diccionario de mapeo para valores en X_test, asignando -1 a valores no vistos
        classes_mapping = {label: idx for idx, label in enumerate(le.classes_)}
        X_test[col] = X_test[col].astype(str).map(classes_mapping).fillna(-1).astype(int)

    # Imputar columnas numéricas en X_train y X_test
    print("Imputando columnas numéricas...")
    numeric_imputer = SimpleImputer(strategy='mean')
    X_train[numeric_columns] = numeric_imputer.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = numeric_imputer.transform(X_test[numeric_columns])

    # Filtro para quedarnos solo con columnas numéricas después del procesamiento
    X_train = X_train.select_dtypes(include=['int64', 'float64'])
    X_test = X_test.select_dtypes(include=['int64', 'float64'])
    X_test = X_test[X_train.columns]

    # Sobremuestreo de clases minoritarias en y_train
    print("Realizando sobremuestreo en clases minoritarias...")
    df_train_balanced = pd.DataFrame()
    y_train_balanced = pd.Series(dtype=y_train.dtype)
    for clase, count in y_train.value_counts().items():
        df_clase = X_train[y_train == clase]
        y_clase = y_train[y_train == clase]
        if count < y_train.value_counts().max():
            df_clase_oversampled, y_clase_oversampled = resample(
                df_clase, y_clase, replace=True, n_samples=y_train.value_counts().max(), random_state=42
            )
            df_train_balanced = pd.concat([df_train_balanced, df_clase_oversampled])
            y_train_balanced = pd.concat([y_train_balanced, y_clase_oversampled])
        else:
            df_train_balanced = pd.concat([df_train_balanced, df_clase])
            y_train_balanced = pd.concat([y_train_balanced, y_clase])

    # Definir el modelo con parámetros fijos
    print("Definiendo el modelo de Decision Tree con parámetros fijos...")
    best_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,         # Profundidad máxima moderada para reducir sobreajuste
        min_samples_split=10, # Mínimo de muestras necesarias para dividir un nodo
        min_samples_leaf=5    # Mínimo de muestras necesarias en cada hoja
    )

    # Entrenar el modelo
    print("Entrenando el modelo...")
    best_model.fit(df_train_balanced, y_train_balanced)
    print("Entrenamiento del modelo completo.")

    # Extraer las 20 características principales por importancia del modelo
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    top_20_features_importance = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

    # Preparar las 20 variables principales por correlación
    top_20_correlated_df = top_20_correlated_vars.reset_index()
    top_20_correlated_df.columns = ['Feature', 'Correlation']

    # Combinar resultados en JSON
    results_json = {
        "top_20_by_model_importance": top_20_features_importance.to_dict(orient="records"),
        "top_20_by_correlation": top_20_correlated_df.to_dict(orient="records")
    }

    return results_json

if __name__ == "__main__":
    import sys

    # Verificar que se haya proporcionado el parámetro desde la línea de comandos
    if len(sys.argv) != 2:
        print("Uso: python script.py <target_variable>")
        sys.exit(1)

    # Obtener la variable objetivo desde los argumentos de la línea de comandos
    target_variable = sys.argv[1]

    # Llamar a la función run_analysis con la variable objetivo
    try:
        result = run_analysis(target_variable)
        print("JSON final de resultados:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error al ejecutar el análisis: {e}")

