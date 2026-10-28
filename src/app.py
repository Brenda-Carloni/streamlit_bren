
import streamlit as st
import xgboost as xgb
import pandas as pd
from pathlib import Path

# Cone stos pasos vamos a configurar Streamlit:

st.set_page_config(
    page_title="Predicción de diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Predicción de diabetes")
st.write(
    "Completá los datos para obtener una predicción "
    "del modelo de aprendizaje automático."
)

st.info(
    "Esta aplicación tiene fines educativos. "
    "La predicción no reemplaza un diagnóstico médico."
)

# Después cargamos el modelo ya entrenado:

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_model.json"

@st.cache_resource
def cargar_modelo():
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(MODEL_PATH))
    return modelo

try:
    model = cargar_modelo()

except FileNotFoundError:
    st.error(
        f"No se encontró el archivo del modelo: {MODEL_PATH}. "
        "Verificá que esté guardado en la carpeta models."
    )
    st.stop()

except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

# Ingresamos los datos de la persona:

st.subheader("Datos de la persona")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Embarazos",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucosa",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Presión arterial diastólica",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Grosor del pliegue cutáneo",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

with col2:
    insulin = st.number_input(
        "Insulina",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "Índice de masa corporal (BMI)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Edad (años)",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

# Y predecimos:

if st.button("Predecir", type="primary"):

    # Los nombres y el orden coinciden con X
    # utilizado durante el entrenamiento.

    datos = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    try:
        prediccion = int(model.predict(datos)[0])

        st.divider()
        st.subheader("Resultado de la predicción")

        if prediccion == 0:
            st.success(
                "El modelo predice: No tiene diabetes."
            )

        elif prediccion == 1:
            st.warning(
                "El modelo predice: Tiene diabetes."
            )

        # Mostrar probabilidades estimadas por el modelo
        if hasattr(model, "predict_proba"):
            probabilidades = model.predict_proba(datos)[0]
            clases = list(model.classes_)

            prob_diabetes = probabilidades[clases.index(1)]

            st.write(
                f"Probabilidad estimada para la clase "
                f"diabetes: **{prob_diabetes:.1%}**"
            )

            st.caption(
                "Esta probabilidad es una estimación del modelo, "
                "no una probabilidad clínica validada."
            )

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
