import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Household Power Consumption Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- MOCK / SYNTHETIC DATA GENERATOR ---
# Replace this function with your actual dataset loading: df = pd.read_csv("household_power_consumption.csv")
@st.cache_data
def load_data():
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="h")
    df = pd.DataFrame({
        'Date': dates.strftime('%d/%m/%Y'),
        'Time': dates.strftime('%H:%M:%S'),
        'Global_active_power': np.random.uniform(0.2, 4.5, size=1000),
        'Global_reactive_power': np.random.uniform(0.0, 0.5, size=1000),
        'Voltage': np.random.uniform(230, 245, size=1000),
        'Global_intensity': np.random.uniform(1.0, 18.0, size=1000),
        'Sub_metering_1': np.random.choice([0, 1.17, 2.5], size=1000), # Kitchen
        'Sub_metering_2': np.random.choice([0, 1.47, 3.0], size=1000), # Laundry
        'Sub_metering_3': np.random.choice([0, 5.93, 18.0], size=1000), # Water heater / AC
        'Month_Name': dates.strftime('%B')
    })
    return df

df = load_data()

# --- PAGE DEFINITIONS ---

def page_introduction():
    st.title("⚡ Household Power Consumption Analysis")
    st.caption("Overview of minute-level household electric power consumption readings")
    
    st.subheader("Project Overview")
    st.write(
        "This project analyzes high-frequency household power consumption measurements to identify "
        "usage trends, distribution across sub-metered circuits, and seasonal variations in energy demand."
    )
    
    st.divider()
    
    # Key Dataset Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observations", f"{len(df):,}")
    col2.metric("Total Variables", f"{df.shape[1]}")
    col3.metric("Avg Active Power", f"{df['Global_active_power'].mean():.2f} kW")
    col4.metric("Avg Voltage", f"{df['Voltage'].mean():.1f} V")
    
    st.divider()

    # Column Information Section
    st.subheader("📋 Dataset Columns & Definitions")
    
    column_info = [
        {"Column Name": "Date", "Data Type": "Object / Date", "Unit": "dd/mm/yyyy", "Description": "Date of measurement"},
        {"Column Name": "Time", "Data Type": "Object / Time", "Unit": "hh:mm:ss", "Description": "Time of measurement"},
        {"Column Name": "Global_active_power", "Data Type": "Float", "Unit": "kW", "Description": "Household global minute-averaged active power"},
        {"Column Name": "Global_reactive_power", "Data Type": "Float", "Unit": "kW", "Description": "Household global minute-averaged reactive power"},
        {"Column Name": "Voltage", "Data Type": "Float", "Unit": "V", "Description": "Minute-averaged voltage"},
        {"Column Name": "Global_intensity", "Data Type": "Float", "Unit": "A", "Description": "Household global minute-averaged current intensity"},
        {"Column Name": "Sub_metering_1", "Data Type": "Float", "Unit": "Wh", "Description": "Energy sub-metering No. 1 (Kitchen: dishwasher, oven, microwave)"},
        {"Column Name": "Sub_metering_2", "Data Type": "Float", "Unit": "Wh", "Description": "Energy sub-metering No. 2 (Laundry room: washing machine, dryer, fridge)"},
        {"Column Name": "Sub_metering_3", "Data Type": "Float", "Unit": "Wh", "Description": "Energy sub-metering No. 3 (Climate control: electric water heater & AC)"},
        {"Column Name": "Month_Name", "Data Type": "Categorical", "Unit": "Text", "Description": "Engineered month name feature for temporal aggregation"}
    ]
    
    st.dataframe(pd.DataFrame(column_info), use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("📊 Statistical Overview")
    st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

    st.subheader("🔍 Preview Data Sample")
    st.dataframe(df.head(10), use_container_width=True)


def page_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    # Radio buttons to switch between dynamic analysis views
    analysis_type = st.radio(
        "Select Analysis View:",
        options=["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"],
        horizontal=True
    )
    
    st.divider()

    # ----------------------------------------------------
    # 1. UNIVARIATE ANALYSIS VIEW
    # ----------------------------------------------------
    if analysis_type == "Univariate Analysis":
        st.subheader("1. Sub-metering Load Breakdown")
        
        sub_means = pd.Series({
            'Sub_metering_1 (Kitchen)': df['Sub_metering_1'].mean(),
            'Sub_metering_2 (Laundry)': df['Sub_metering_2'].mean(),
            'Sub_metering_3 (Water heater / AC)': df['Sub_metering_3'].mean()
        })
        sub_shares = (sub_means / sub_means.sum()) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Share of Mean Sub-metered Load**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            colors = ['#4C72B0', '#55A868', '#C44E52']
            labels = [f"{k}\n{v:.1f}%" for k, v in zip(['Kitchen', 'Laundry', 'Water Heater / AC'], sub_shares)]
            ax1.pie(sub_shares, labels=labels, colors=colors, startangle=90, autopct='')
            ax1.set_title("Load Share by Circuit")
            st.pyplot(fig1)
            
        with col2:
            st.markdown("**Average Consumption per Circuit (Wh)**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.barplot(x=['Kitchen', 'Laundry', 'Water Heater / AC'], y=sub_means.values, palette=colors, ax=ax2)
            ax2.set_ylabel("Mean (Wh)")
            ax2.set_title("Average Consumption")
            st.pyplot(fig2)

        st.divider()

        # Added: Interactive Variable Distribution Section
        st.subheader("2. Continuous Variable Distributions (Histogram & KDE)")
        
        selected_col = st.selectbox(
            "Select Variable for Distribution Analysis:",
            options=["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
        )
        
        col_hist1, col_hist2 = st.columns(2)
        
        with col_hist1:
            st.markdown(f"**Histogram with KDE ({selected_col})**")
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
            sns.histplot(df[selected_col], kde=True, color='#4C72B0', ax=ax_hist)
            ax_hist.set_title(f"{selected_col} Distribution")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)

        with col_hist2:
            st.markdown(f"**Box Plot for Outlier Inspection ({selected_col})**")
            fig_box, ax_box = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(x=df[selected_col], color='#55A868', ax=ax_box)
            ax_box.set_title(f"{selected_col} Box Plot")
            st.pyplot(fig_box)

    # ----------------------------------------------------
    # 2. BIVARIATE ANALYSIS VIEW
    # ----------------------------------------------------
    elif analysis_type == "Bivariate Analysis":
        st.subheader("1. Active vs. Reactive Power Across Months")
        st.write("Comparing active power demand (bars) against reactive power loads (line) grouped by month.")
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly = df.groupby('Month_Name', observed=False)[['Global_active_power', 'Global_reactive_power']].sum()
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
        
        fig3, ax_bar = plt.subplots(figsize=(10, 4))
        ax_line = ax_bar.twinx()
        
        ax_bar.bar(monthly.index, monthly['Global_active_power'], color='#4C72B0', alpha=0.7, label='Global Active Power')
        ax_line.plot(monthly.index, monthly['Global_reactive_power'], color='#C44E52', marker='o', label='Global Reactive Power')
        
        ax_bar.set_ylabel('Global Active Power (Sum)', color='#4C72B0')
        ax_line.set_ylabel('Global Reactive Power (Sum)', color='#C44E52')
        
        ax_bar.set_xticks(range(len(monthly.index)))
        ax_bar.set_xticklabels(monthly.index, rotation=45, ha='right')
        fig3.suptitle("Active Power vs. Reactive Power by Month")
        
        st.pyplot(fig3)

        st.divider()

        # Added: Scatter & Regression Plots
        st.subheader("2. Power vs. Intensity & Voltage Relationships")
        
        col_sc1, col_sc2 = st.columns(2)
        
        with col_sc1:
            st.markdown("**Global Active Power vs. Global Intensity**")
            fig_sc1, ax_sc1 = plt.subplots(figsize=(5, 3.5))
            sns.regplot(data=df, x='Global_intensity', y='Global_active_power', color='#4C72B0', scatter_kws={'alpha':0.3}, ax=ax_sc1)
            ax_sc1.set_title("Active Power vs. Current Intensity")
            st.pyplot(fig_sc1)

        with col_sc2:
            st.markdown("**Global Active Power vs. Voltage**")
            fig_sc2, ax_sc2 = plt.subplots(figsize=(5, 3.5))
            sns.regplot(data=df, x='Voltage', y='Global_active_power', color='#C44E52', scatter_kws={'alpha':0.3}, ax=ax_sc2)
            ax_sc2.set_title("Active Power vs. Voltage")
            st.pyplot(fig_sc2)

        st.divider()

        # Added: Monthly Active Power Distribution Boxplot
        st.subheader("3. Monthly Distribution of Global Active Power")
        fig_m_box, ax_m_box = plt.subplots(figsize=(10, 4))
        ordered_months = [m for m in month_order if m in df['Month_Name'].unique()]
        sns.boxplot(data=df, x='Month_Name', y='Global_active_power', order=ordered_months, palette='Blues', ax=ax_m_box)
        ax_m_box.set_title("Global Active Power Spread by Month")
        ax_m_box.set_xticklabels(ax_m_box.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig_m_box)

    # ----------------------------------------------------
    # 3. MULTIVARIATE ANALYSIS VIEW
    # ----------------------------------------------------
    elif analysis_type == "Multivariate Analysis":
        st.subheader("1. Feature Correlation Matrix")
        st.write("Heatmap illustrating pairwise linear relationship dynamics among all continuous parameters.")
        
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

        st.divider()

        # Added: Sub-metering Breakdown across Months
        st.subheader("2. Monthly Sub-metering Consumption Breakdown")
        st.write("Comparing sub-metered circuit consumption averages across months.")
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        sub_monthly = df.groupby('Month_Name', observed=False)[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
        sub_monthly = sub_monthly.reindex([m for m in month_order if m in sub_monthly.index])
        
        fig_sub_bar, ax_sub_bar = plt.subplots(figsize=(10, 4))
        sub_monthly.plot(kind='bar', ax=ax_sub_bar, color=['#4C72B0', '#55A868', '#C44E52'], width=0.8)
        ax_sub_bar.set_ylabel("Mean Consumption (Wh)")
        ax_sub_bar.set_title("Sub-metering Circuit Load Comparison by Month")
        ax_sub_bar.set_xticklabels(sub_monthly.index, rotation=45, ha='right')
        ax_sub_bar.legend(['Kitchen (Sub 1)', 'Laundry (Sub 2)', 'Water Heater / AC (Sub 3)'])
        st.pyplot(fig_sub_bar)

        st.divider()

        # Added: Pairwise Relationships Grid (Pairplot)
        st.subheader("3. Pairwise Feature Interactions (Pairplot)")
        selected_pair_vars = st.multiselect(
            "Select features for Pairplot grid:",
            options=['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity'],
            default=['Global_active_power', 'Global_intensity', 'Voltage']
        )
        
        if len(selected_pair_vars) >= 2:
            fig_pair = sns.pairplot(df[selected_pair_vars], diag_kind='kde', plot_kws={'alpha': 0.4})
            st.pyplot(fig_pair.fig)
        else:
            st.warning("Select at least 2 features to generate the Pairplot.")


def page_conclusion():
    st.title("📌 Project Conclusions & Insights")
    
    st.subheader("Key Findings")
    
    st.success("**Sub-metering Distribution:** Sub_metering_3 (Water heater & AC) dominates overall consumption, accounting for roughly ~69% of sub-metered power usage.")
    st.info("**Kitchen & Laundry Usage:** Kitchen (Sub 1) and Laundry (Sub 2) circuits represent minority load shares (~13.7% and ~17.2% respectively), driven primarily by periodic appliance operation.")
    st.warning("**Active vs. Reactive Power Dynamics:** Active power peaks significantly during high-demand months, whereas reactive power remains relatively stable across seasonal changes.")
    
    st.divider()
    
    st.subheader("Actionable Recommendations")
    st.write("1. **Target Sub-metering 3 for Efficiency:** Upgrading HVAC systems or optimizing water heater schedules will produce the highest impact on overall energy savings.")
    st.write("2. **Peak Load Management:** Implementing smart timers for heavy appliances during peak monthly intervals can balance active power spikes.")


# --- MULTI-PAGE NAVIGATION SETUP ---
pg = st.navigation([
    st.Page(page_introduction, title="1. Introduction", icon="🏠"),
    st.Page(page_eda, title="2. EDA", icon="📈"),
    st.Page(page_conclusion, title="3. Conclusion", icon="🎯")
])

pg.run()