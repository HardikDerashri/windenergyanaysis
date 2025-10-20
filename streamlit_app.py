# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Wind Energy Viability Tool (Assignment)",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants and Estimates ---
COST_MULTIPLIER_CAPEX = 1500.0  # Base CAPEX estimate (€/kW)
COST_MULTIPLIER_OPEX = 50.0     # Base OPEX estimate (€/kW/year)
HOURS_PER_YEAR = 8760           # Hours in a year

# --- Core Calculation Functions ---

def calculate_FCR(interest_rate, project_life):
    """
    Step 3: Calculates the Capital Recovery Factor (FCR).
    FCR = [i * (1 + i)^N] / [(1 + i)^N - 1]
    """
    i = interest_rate / 100
    if i <= 0:
        return 1 / project_life if project_life > 0 else 0
    
    # Use numpy to avoid potential OverflowError with large exponents
    try:
        fcr = (i * np.power(1 + i, project_life)) / (np.power(1 + i, project_life) - 1)
        return fcr
    except Exception:
        return 1.0 

def estimate_AEP(P_rated_MW, V_avg_m_s):
    """
    Step 2: Estimates Annual Energy Production (AEP) and Capacity Factor (CF).
    A simple heuristic model for CF based on average wind speed (V_avg) is used.
    AEP (MWh) = P_rated (MW) * Hours/Year * CF
    """
    # Heuristic CF Model: Penalizes low speeds, caps at ~55%
    CF = 0.006 * (V_avg_m_s**2) + 0.01 * V_avg_m_s + 0.1
    CF = max(0.1, min(0.55, CF)) 

    AEP_MWh = P_rated_MW * HOURS_PER_YEAR * CF
    AEP_GWh = AEP_MWh / 1000
    
    return AEP_MWh, AEP_GWh, CF

def calculate_LCOE(CAPEX_total, OPEX_annual, AEP_MWh, FCR):
    """
    Step 3: Calculates the Levelized Cost of Energy (LCOE) in €/MWh.
    LCOE = [FCR * CAPEX_total + OPEX_annual] / AEP_MWh
    """
    if AEP_MWh <= 0:
        return float('inf')
        
    LCOE = (FCR * CAPEX_total + OPEX_annual) / AEP_MWh
    return LCOE

# ----------------------------------------------------
# --- SIDEBAR (INPUTS) ---
# ----------------------------------------------------

st.sidebar.title('⚙️ Wind Project Parameters')

# ----------------------------------------------------
# STEP 1: Site and Wind Resource
# ----------------------------------------------------
st.sidebar.header('1. Wind Resource (Location & V_avg)')
# This text input serves as the location selector (API implementation is not needed for the core calculation)
location = st.sidebar.text_input('Location Name', 'Tarifa, Spain') 
# THE CRUCIAL INPUT for Step 1 & 2
V_avg_m_s = st.sidebar.slider('Average Wind Speed ($V_{avg}$ in m/s)', 4.0, 12.0, 7.5, 0.1, help="Look up this value from a wind atlas for your chosen location and hub height.")
hub_height_m = st.sidebar.slider('Hub Height (m)', 60, 160, 120, 5)

# ----------------------------------------------------
# STEP 2: Turbine Specifications
# ----------------------------------------------------
st.sidebar.header('2. Turbine Characteristics (Energy)')
P_rated_MW = st.sidebar.number_input('Rated Power ($P_{rated}$ in MW)', min_value=1.0, max_value=10.0, value=3.0, step=0.1)
rotor_diameter_m = st.sidebar.slider('Rotor Diameter (m)', 80, 200, 130, 5)

# ----------------------------------------------------
# STEP 3: Costs and Finance (LCOE)
# ----------------------------------------------------
st.sidebar.header('3. Costs and Finance (LCOE)')
capex_per_kW = st.sidebar.number_input('CAPEX Specific (€/kW)', value=COST_MULTIPLIER_CAPEX, step=50.0, help="Initial investment cost per kW installed.")
opex_per_kW_yr = st.sidebar.number_input('OPEX Specific (€/kW/year)', value=COST_MULTIPLIER_OPEX, step=5.0, help="Annual O&M cost per kW.")
project_life_years = st.sidebar.slider('Project Lifetime (Years)', 15, 30, 25, 1)
interest_rate = st.sidebar.slider('Nominal Annual Interest Rate (%)', 1.0, 10.0, 5.0, 0.5, help="Financial cost for calculating the FCR.")

# ----------------------------------------------------
# STEP 4: Economic Viability (Market)
# ----------------------------------------------------
st.sidebar.header('4. Market and Viability')
market_price_MWh = st.sidebar.number_input('Market Selling Price (€/MWh)', min_value=20.0, max_value=150.0, value=65.0, step=1.0)

st.sidebar.markdown("---")

# ----------------------------------------------------
# --- CENTRAL CALCULATION LOGIC ---
# ----------------------------------------------------

P_rated_kW = P_rated_MW * 1000

# 1. & 2. AEP Calculation
AEP_MWh, AEP_GWh, CF = estimate_AEP(P_rated_MW, V_avg_m_s)

# 3. Cost Calculation (CAPEX, OPEX, FCR)
CAPEX_total = P_rated_kW * capex_per_kW
OPEX_annual = P_rated_kW * opex_per_kW_yr
FCR = calculate_FCR(interest_rate, project_life_years)

if AEP_MWh > 0:
    LCOE = calculate_LCOE(CAPEX_total, OPEX_annual, AEP_MWh, FCR)
else:
    LCOE = float('inf')

# 4. Viability Analysis
annual_income = AEP_MWh * market_price_MWh
annual_cost = FCR * CAPEX_total + OPEX_annual
profit_loss = annual_income - annual_cost
is_viable = LCOE <= market_price_MWh
subsidy_needed_MWh = max(0, LCOE - market_price_MWh)
subsidy_total_annual = subsidy_needed_MWh * AEP_MWh

# ----------------------------------------------------
# --- DASHBOARD (RESULTS) ---
# ----------------------------------------------------

st.title("🌱 Wind Farm Economic Viability Analysis")
st.subheader(f"Project: {P_rated_MW:.1f} MW at {location}")
st.markdown("---")

# Global KPIs (Steps 2, 3, and 4)
st.header("Global Economic Indicators")

kpi1, kpi2, kpi3 = st.columns(3)

# KPI 1: LCOE (Step 3)
kpi1.metric(
    "LCOE (Levelized Cost of Energy)",
    f"€ {LCOE:,.2f} /MWh",
    f" vs Market: € {market_price_MWh:,.2f}",
    delta_color="off" if is_viable else "inverse"
)

# KPI 2: AEP (Step 2)
kpi2.metric("Annual Energy Production (AEP)", f"{AEP_GWh:,.1f} GWh")

# KPI 3: Annual Profit/Loss (Step 4)
profit_color = "inverse" if profit_loss < 0 else "normal"
kpi3.metric("Annual Profit / Loss", f"€ {profit_loss:,.0f}", delta_color=profit_color)

# Viability Status (Step 4)
viability_status = "VIABLE" if is_viable else "NOT VIABLE"
st.markdown(f"""
<div style="border: 2px solid {'green' if is_viable else 'red'}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px; font-size: 20px;">
    ECONOMIC STATUS: {viability_status}
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Detailed Breakdown
col_steps, col_charts = st.columns([1, 1], gap="large")

with col_steps:
    st.subheader("Analysis Breakdown (Steps 1-4)")
    
    # Detail for Step 1 & 2
    st.markdown("### 🗺️ Step 1 & 2: Resource and Energy")
    st.markdown(f"**Location:** `{location}`")
    st.metric("Average Wind Speed", f"{V_avg_m_s} m/s", help="Key input from wind atlas.")
    st.metric("Capacity Factor (CF)", f"{CF * 100:.1f} %")
    st.metric("Rated Power", f"{P_rated_MW:.1f} MW")

    # Detail for Step 3
    st.markdown("### 💰 Step 3: Costs and LCOE")
    
    # DataFrame for Cost Table
    cost_data = {
        'CAPEX Total (Initial Investment)': CAPEX_total,
        'OPEX Annual (Operations)': OPEX_annual,
        'Annual Financial Cost (FCR * CAPEX)': FCR * CAPEX_total
    }
    df_costs = pd.DataFrame(cost_data.items(), columns=['Component', 'Cost (€)'])
    df_costs['Cost (€)'] = df_costs['Cost (€)'].round(0).map('{:,.0f}'.format)
    
    st.dataframe(df_costs, use_container_width=True, hide_index=True)
    st.metric("Capital Recovery Factor (FCR)", f"{FCR * 100:.2f} %")
    st.metric("Total Annual Cost", f"€ {annual_cost:,.0f}")

    # Detail for Step 4
    st.markdown("### 💸 Step 4: Viability and Subsidy")
    st.metric("Estimated Annual Income", f"€ {annual_income:,.0f}")
    
    if not is_viable:
        st.error(f"Subsidy required: € **{subsidy_needed_MWh:,.2f} /MWh** or € **{subsidy_total_annual:,.0f}** annually to break even.")
    else:
        st.success("The project is economically viable without additional subsidies.")

with col_charts:
    st.subheader("Economic Model Visualization")
    
    # Gauge Chart: LCOE vs Market Price
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=LCOE,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "LCOE vs. Market Price (€/MWh)"},
        delta={'reference': market_price_MWh, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max(market_price_MWh * 1.5, LCOE * 1.2)]},
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
    
    # Bar Chart: Annual Income vs Cost
    df_annual_breakdown = pd.DataFrame({
        'Type': ['Annual Income', 'Annual Cost'],
        'Value (€)': [annual_income, annual_cost]
    })

    fig_bar = px.bar(df_annual_breakdown, x='Type', y='Value (€)', 
                     color='Type', 
                     title='Annual Income vs. Annual Cost',
                     color_discrete_map={'Annual Income': 'green', 'Annual Cost': 'red'})
    fig_bar.update_traces(marker_line_width=0)
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
