import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Segmentación Comercial - ML App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Segmentación Comercial de Clientes con Machine Learning")
st.markdown("Esta aplicación analiza las transacciones comerciales, agrupa a los clientes utilizando **K-Means Clustering** y permite simular la clasificación de nuevos clientes en tiempo real.")

# ---------------------------------------------------------
# Carga y Preprocesamiento de Datos
# ---------------------------------------------------------
@st.cache_data
def cargar_y_preprocesar():
    # Cargar base de datos
    df = pd.read_excel('BASE DE DATOS ACA1.xlsx')
    
    # Métrica calculada: Monto Total
    df['Monto Total (£)'] = df['Cantidad'] * df['Precio Unitario (£)']
    
    # Consolidación a nivel de cliente (RFM simplificado)
    clientes_df = df.groupby('ID de Cliente').agg({
        'Nº Factura': 'count',
        'Monto Total (£)': 'sum',
        'Cantidad': 'mean'
    }).rename(columns={
        'Nº Factura': 'Frecuencia',
        'Monto Total (£)': 'Gasto_Total',
        'Cantidad': 'Cantidad_Promedio'
    }).reset_index()
    
    # Entrenar K-Means con K=2
    features = ['Frecuencia', 'Gasto_Total', 'Cantidad_Promedio']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clientes_df[features])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clientes_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Mapear nombres a los clústeres
    cluster_names = {
        0: 'Clúster 0: Minorista Recurrente',
        1: 'Clúster 1: Mayorista VIP'
    }
    clientes_df['Nombre_Cluster'] = clientes_df['Cluster'].map(cluster_names)
    
    return df, clientes_df, scaler, kmeans

try:
    df_raw, clientes_df, scaler, kmeans = cargar_y_preprocesar()
except Exception as e:
    st.error(f"Error al cargar el archivo de datos: {e}")
    st.info("Asegúrate de colocar 'BASE DE DATOS ACA1.xlsx' en la misma carpeta que app.py")
    st.stop()

# ---------------------------------------------------------
# Sidebar / Panel Lateral
# ---------------------------------------------------------
st.sidebar.header("🎛️ Filtros y Controles")

pais_filtro = st.sidebar.multiselect(
    "Filtrar por País:",
    options=df_raw['País'].unique(),
    default=df_raw['País'].unique()
)

df_filtrado = df_raw[df_raw['País'].isin(pais_filtro)]

# ---------------------------------------------------------
# KPIs y Métricas Clave
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Ventas (£)", f"£{df_filtrado['Monto Total (£)'].sum():,.2f}")
with col2:
    st.metric("Transacciones", len(df_filtrado))
with col3:
    st.metric("Clientes Únicos", clientes_df['ID de Cliente'].nunique())
with col4:
    st.metric("Segmentos ML", "2 Clústeres")

st.markdown("---")

# ---------------------------------------------------------
# Visualización Interactiva del Modelo
# ---------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Mapa de Segmentación (Gasto vs. Cantidad Promedio)")
    
    fig = px.scatter(
        clientes_df,
        x='Gasto_Total',
        y='Cantidad_Promedio',
        color='Nombre_Cluster',
        size='Frecuencia',
        hover_data=['ID de Cliente'],
        labels={
            'Gasto_Total': 'Gasto Total (£)',
            'Cantidad_Promedio': 'Cantidad Promedio por Pedido',
            'Nombre_Cluster': 'Segmento'
        },
        title="Clústeres K-Means por Cliente",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📊 Resumen por Segmento")
    resumen_cluster = clientes_df.groupby('Nombre_Cluster')[['Frecuencia', 'Gasto_Total', 'Cantidad_Promedio']].mean().round(2)
    st.dataframe(resumen_cluster, use_container_width=True)
    
    st.markdown("""
    **Interpretación Comercial:**
    * **Minorista Recurrente:** Compras más frecuentes, menor volumen por pedido.
    * **Mayorista VIP:** Menor frecuencia, pero volúmenes y gastos acumulados significativamente mayores.
    """)

st.markdown("---")

# ---------------------------------------------------------
# Simulador de Clasificación de Nuevos Clientes
# ---------------------------------------------------------
st.subheader("🧮 Simulador / Clasificador de Nuevo Cliente")
st.write("Ingresa los datos del nuevo cliente para determinar su clúster comercial mediante el modelo K-Means:")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    frec_input = st.number_input("Frecuencia (Compras registradas)", min_value=1, value=10)
with col_s2:
    gasto_input = st.number_input("Gasto Total Acumulado (£)", min_value=1.0, value=250.0)
with col_s3:
    cant_input = st.number_input("Cantidad Promedio por Pedido", min_value=1.0, value=8.0)

if st.button("🚀 Clasificar Cliente"):
    nuevo_dato = np.array([[frec_input, gasto_input, cant_input]])
    nuevo_dato_scaled = scaler.transform(nuevo_dato)
    prediccion = kmeans.predict(nuevo_dato_scaled)[0]
    
    if prediccion == 1:
        st.success(f"🎉 **El cliente pertenece al Clúster 1: Mayorista VIP**")
        st.info("👉 **Estrategia sugerida:** Ofrecer atención personalizada, ofertas por volumen y canal de soporte prioritario.")
    else:
        st.success(f"🔹 **El cliente pertenece al Clúster 0: Minorista Recurrente**")
        st.info("👉 **Estrategia sugerida:** Implementar campañas de venta cruzada (*cross-selling*) e incentivos de acumulación de puntos.")

# ---------------------------------------------------------
# Mostrar Tabla Completa de Datos
# ---------------------------------------------------------
with st.expander("🔍 Ver Tabla de Datos Consolidados por Cliente"):
    st.dataframe(clientes_df, use_container_width=True)
