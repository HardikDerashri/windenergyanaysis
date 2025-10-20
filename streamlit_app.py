# Importar librerías
import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px

# --- Configuración de la página ---
st.set_page_config(
    page_title="Herramienta de Viabilidad Eólica (Asignación)",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes y Mapeos Ilustrativos ---
# Simplificaciones para los cálculos.
COST_MULTIPLIER_CAPEX = 1500.0  # CAPEX base estimado (€/kW)
COST_MULTIPLIER_OPEX = 50.0     # OPEX base estimado (€/kW/año)
HOURS_PER_YEAR = 8760           # Horas en un año

# --- Funciones de Cálculo para la Asignación ---

def calculate_FCR(interest_rate, project_life):
    """
    Paso 3: Calcula el Factor de Recuperación de Capital (FCR), simplificando el costo financiero.
    FCR = [i * (1 + i)^N] / [(1 + i)^N - 1]
    """
    i = interest_rate / 100
    if i == 0:
        return 1 / project_life
    
    # Asume un valor si el denominador es 0, aunque es poco probable con i > 0
    if (1 + i)**project_life - 1 == 0:
        return 1.0 
        
    return (i * (1 + i)**project_life) / ((1 + i)**project_life - 1)

def estimate_AEP(P_rated_MW, V_avg_m_s, P_rated_kW):
    """
    Paso 2: Estima la Producción Anual de Energía (AEP) y el Factor de Capacidad (CF).
    Utiliza una función heurística simple para el Factor de Capacidad (CF) basada en V_avg.
    AEP (MWh) = P_rated (MW) * Horas/Año * CF
    """
    # Modelo heurístico simple para CF (ejemplo: 6 m/s -> CF ~0.3; 9 m/s -> CF ~0.5)
    # Penaliza velocidades bajas y se satura a velocidades altas.
    CF = 0.006 * (V_avg_m_s**2) + 0.01 * V_avg_m_s + 0.1
    CF = max(0.1, min(0.55, CF)) # Limitar CF entre 10% y 55%

    AEP_MWh = P_rated_MW * HOURS_PER_YEAR * CF
    
    # Cálculo de la Potencia Teórica Máxima (Ley de Betz) para comparación
    # Aunque no se usa en AEP, es útil para el análisis.
    # Área del rotor (m²)
    # rotor_area = math.pi * (rotor_diameter_m / 2)**2
    # P_max_theoretical_kW = 0.5 * air_density * rotor_area * (V_avg_m_s**3) * 0.593 / 1000
    
    # AEP (GWh)
    AEP_GWh = AEP_MWh / 1000
    
    return AEP_MWh, AEP_GWh, CF

def calculate_LCOE(CAPEX_total, OPEX_annual, AEP_MWh, FCR):
    """
    Paso 3: Calcula el Coste Nivelado de la Energía (LCOE) en €/MWh.
    LCOE = [FCR * CAPEX_total + OPEX_annual] / AEP_MWh
    """
    if AEP_MWh <= 0:
        return float('inf')
        
    LCOE = (FCR * CAPEX_total + OPEX_annual) / AEP_MWh
    return LCOE

# --- BARRA LATERAL (INPUTS DE LA ASIGNACIÓN) ---
st.sidebar.title('⚙️ Parámetros del Proyecto Eólico')

# ----------------------------------------------------
# PASO 1: Emplazamiento y Recurso Eólico (Recurso)
# ----------------------------------------------------
st.sidebar.header('1. Recurso Eólico (Ubicación)')
location = st.sidebar.text_input('Ubicación (Ej: Tarifa, España)', 'Tarifa, Spain')
V_avg_m_s = st.sidebar.slider('Velocidad Media del Viento ($V_{avg}$ en m/s)', 4.0, 12.0, 7.5, 0.1, help="Dato clave del Atlas Eólico a la altura del buje.")
hub_height_m = st.sidebar.slider('Altura del Buje (Hub Height en m)', 60, 160, 120, 5, help="Altura del centro del rotor sobre el suelo.")
# air_density = st.sidebar.number_input('Densidad del Aire (kg/m³)', value=1.225, step=0.005)

# ----------------------------------------------------
# PASO 2: Especificaciones de la Turbina (Energía)
# ----------------------------------------------------
st.sidebar.header('2. Características de la Turbina')
P_rated_MW = st.sidebar.number_input('Potencia Nominal ($P_{rated}$ en MW)', min_value=1.0, max_value=10.0, value=3.0, step=0.1, help="Potencia máxima del generador.")
rotor_diameter_m = st.sidebar.slider('Diámetro del Rotor (D en m)', 80, 200, 130, 5, help="Determina el área de captación del viento.")

# ----------------------------------------------------
# PASO 3: Costes y Finanzas (Costos)
# ----------------------------------------------------
st.sidebar.header('3. Costes y Finanzas (LCOE)')
capex_per_kW = st.sidebar.number_input('CAPEX Específico (€/kW)', value=COST_MULTIPLIER_CAPEX, step=50.0, help="Costo de inversión inicial por kW instalado (Turbina + BOS + Otros).")
opex_per_kW_yr = st.sidebar.number_input('OPEX Específico (€/kW/año)', value=COST_MULTIPLIER_OPEX, step=5.0, help="Costo Operacional y de Mantenimiento por kW por año.")
project_life_years = st.sidebar.slider('Vida Útil del Proyecto (Años)', 15, 30, 25, 1)
interest_rate = st.sidebar.slider('Tasa de Interés Anual Nominal (%)', 1.0, 10.0, 5.0, 0.5, help="Costo financiero para calcular el FCR.")

# ----------------------------------------------------
# PASO 4: Viabilidad Económica (Mercado)
# ----------------------------------------------------
st.sidebar.header('4. Mercado y Viabilidad')
market_price_MWh = st.sidebar.number_input('Precio de Venta en Mercado (€/MWh)', min_value=20.0, max_value=150.0, value=65.0, step=1.0, help="Precio de la electricidad de OMIE o equivalente.")

st.sidebar.markdown("---")

# ----------------------------------------------------
# --- LÓGICA DE CÁLCULO CENTRAL ---
# ----------------------------------------------------

# Preparación
P_rated_kW = P_rated_MW * 1000

# 1. & 2. CÁLCULO DE AEP
AEP_MWh, AEP_GWh, CF = estimate_AEP(P_rated_MW, V_avg_m_s, P_rated_kW)

# 3. CÁLCULO DE COSTOS
CAPEX_total = P_rated_kW * capex_per_kW
OPEX_annual = P_rated_kW * opex_per_kW_yr
FCR = calculate_FCR(interest_rate, project_life_years)

if AEP_MWh > 0:
    LCOE = calculate_LCOE(CAPEX_total, OPEX_annual, AEP_MWh, FCR)
else:
    LCOE = float('inf')

# 4. ANÁLISIS DE VIABILIDAD
annual_income = AEP_MWh * market_price_MWh
annual_cost = FCR * CAPEX_total + OPEX_annual
profit_loss = annual_income - annual_cost
is_viable = LCOE <= market_price_MWh
subsidy_needed_MWh = max(0, LCOE - market_price_MWh)
subsidy_total_annual = subsidy_needed_MWh * AEP_MWh

# ----------------------------------------------------
# --- DASHBOARD (RESULTADOS) ---
# ----------------------------------------------------

st.title("🌱 Análisis de Viabilidad Económica Eólica")
st.subheader(f"Proyecto de {P_rated_MW:.1f} MW en {location}")
st.markdown("---")

# ----------------------------------------
# KPIs Globales (Pasos 2, 3 y 4)
# ----------------------------------------
st.header("Resultados Globales de Viabilidad")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# KPI 1: LCOE (Paso 3)
kpi1.metric(
    "LCOE (Coste Nivelado de la Energía)",
    f"€ {LCOE:,.2f} /MWh",
    f" vs Mercado: € {market_price_MWh:,.2f}",
    delta_color="off" if is_viable else "inverse"
)

# KPI 2: AEP (Paso 2)
kpi2.metric("Producción Anual de Energía (AEP)", f"{AEP_GWh:,.1f} GWh")

# KPI 3: Beneficio/Pérdida Anual (Paso 4)
profit_color = "inverse" if profit_loss < 0 else "normal"
kpi3.metric("Beneficio / Pérdida Anual", f"€ {profit_loss:,.0f}", delta_color=profit_color)

# KPI 4: Viabilidad (Paso 4)
viability_status = "VIABLE" if is_viable else "NO VIABLE"
st.markdown(f"""
<div style="border: 2px solid {'green' if is_viable else 'red'}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px;">
    ESTADO ECONÓMICO: {viability_status}
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------
# Desglose Detallado por Pasos
# ----------------------------------------

col_steps, col_charts = st.columns([1, 1], gap="large")

with col_steps:
    st.subheader("Desglose del Análisis (Pasos 1-4)")
    
    # Detalle del Paso 1 & 2
    st.markdown("### 🗺️ Paso 1 & 2: Recurso y Energía (AEP)")
    st.markdown(f"**Ubicación:** `{location}`")
    st.metric("Velocidad Media del Viento", f"{V_avg_m_s} m/s", help="Dato utilizado para la estimación de rendimiento.")
    st.metric("Factor de Capacidad (CF)", f"{CF * 100:.1f} %", help="Fracción de la potencia máxima que se produce en un año.")
    st.metric("Potencia Nominal", f"{P_rated_MW:.1f} MW")

    # Detalle del Paso 3
    st.markdown("### 💰 Paso 3: Costos y LCOE")
    
    # Creación de DataFrame para la tabla de costos
    cost_data = {
        'CAPEX Total (Inversión Inicial)': CAPEX_total,
        'OPEX Anual (Operación)': OPEX_annual,
        'Costo Financiero Anual (FCR * CAPEX)': FCR * CAPEX_total
    }
    df_costs = pd.DataFrame(cost_data.items(), columns=['Componente', 'Costo (€)'])
    df_costs['Costo (€)'] = df_costs['Costo (€)'].round(0).map('{:,.0f}'.format)
    
    st.dataframe(df_costs, use_container_width=True, hide_index=True)
    st.metric("Factor de Recuperación de Capital (FCR)", f"{FCR * 100:.2f} %", help="Tasa anual que representa el retorno de la inversión y el costo del capital.")
    st.metric("Costo Anual Total", f"€ {annual_cost:,.0f}", help="Costo anualizado de la inversión (CAPEX) + costos operativos (OPEX).")

    # Detalle del Paso 4
    st.markdown("### 💸 Paso 4: Viabilidad y Subsidio")
    st.metric("Ingreso Anual Estimado", f"€ {annual_income:,.0f}")
    if not is_viable:
        st.error(f"Se requiere un subsidio de € {subsidy_needed_MWh:,.2f} /MWh o € {subsidy_total_annual:,.0f} anuales para alcanzar la viabilidad.")
    else:
        st.success("El proyecto es económicamente viable sin subsidios adicionales.")

with col_charts:
    st.subheader("Visualización del Modelo Económico")
    
    # Gráfico de Viabilidad (Gauge/Comparación LCOE vs Precio)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=LCOE,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "LCOE vs. Precio de Mercado (€/MWh)"},
        delta={'reference': market_price_MWh, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max(market_price_MWh * 1.5, LCOE * 1.2)], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, market_price_MWh], 'color': "lightgreen"},
                {'range': [market_price_MWh, max(market_price_MWh * 1.5, LCOE * 1.2)], 'color': "lightcoral"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': market_price_MWh}
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(t=50, b=0, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Gráfico de Desglose de Costo Anual
    df_annual_breakdown = pd.DataFrame({
        'Tipo': ['Ingreso Anual', 'Costo Anual'],
        'Valor (€)': [annual_income, annual_cost]
    })

    fig_bar = px.bar(df_annual_breakdown, x='Tipo', y='Valor (€)', 
                     color='Tipo', 
                     title='Ingreso vs. Costo Anual Total',
                     color_discrete_map={'Ingreso Anual': 'green', 'Costo Anual': 'red'})
    fig_bar.update_traces(marker_line_width=0)
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------
# Tareas de Asignación (Resumen de pasos)
# ----------------------------------------
st.markdown("---")
st.subheader("Resumen de las Tareas de la Asignación")
st.info("""
Esta herramienta te permite cumplir con los 4 pasos clave de la asignación:
1. **Elegir y Cuantificar el Recurso:** Seleccionas la ubicación (ej. Tarifa) y la **Velocidad Media del Viento ($V_{avg}$)** en la barra lateral.
2. **Cuantificar la Energía (AEP):** El código calcula el **Factor de Capacidad (CF)** y la **Producción Anual de Energía (AEP)** en GWh/MWh.
3. **Modelar y Calcular Costos (LCOE):** El código utiliza el **CAPEX** y **OPEX** específicos, junto con la **Tasa de Interés** y la **Vida Útil** para obtener el **LCOE**.
4. **Analizar la Viabilidad Económica:** El código compara el **LCOE** con el **Precio de Venta en Mercado** para determinar el **Beneficio/Pérdida Anual** y el posible **Subsidio Necesario**.
""")
