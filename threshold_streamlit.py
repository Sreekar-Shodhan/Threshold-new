import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import os
from indicator_analyzer import IndicatorAnalyzer

# Set page config
st.set_page_config(
    page_title="Development Threshold Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-bq {
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-title">🌍 Development Threshold Analyzer</h1>', unsafe_allow_html=True)
st.markdown("""
    Analyze development indicators across countries and time. 
    Explore when countries reached specific development thresholds and compare their progress.
""")

# Initialize the analyzer
try:
    # Add debugging information
    st.sidebar.expander("🔍 Debug Information", expanded=False).write(
        f"Working Directory: {os.getcwd()}\n"
        f"Data Directory Path: {Path('DATACSV').absolute()}\n"
        f"Data Directory Exists: {Path('DATACSV').exists()}\n"
        f"Files in Data Directory: {[f.name for f in Path('DATACSV').glob('*') if f.is_file()]}\n"
    )
    
    analyzer = IndicatorAnalyzer()
    available_indicators = analyzer.indicators
    
    # Show more debug info after analyzer is initialized
    st.sidebar.expander("🔍 Analyzer Debug Info", expanded=False).write(
        f"Indicators Found: {list(available_indicators.keys())}\n"
        f"Indicator Files: {list(available_indicators.values())}\n"
    )
    
    if not available_indicators:
        st.error("No indicator data found. Please check if the DATACSV directory contains valid CSV files.")
        st.stop()
except Exception as e:
    st.error(f"Error initializing analyzer: {str(e)}")
    # Add error details for debugging
    st.error(f"Error details: {type(e).__name__}")
    import traceback
    st.code(traceback.format_exc(), language="python")
    st.stop()

# Cache the indicator data
@st.cache_data(ttl=3600)
def get_indicators():
    return list(analyzer.indicators.keys())

# Sidebar for controls
with st.sidebar:
    st.header("Analysis Settings")
    
    # Indicator selection
    selected_indicator = st.selectbox(
        "Select an indicator:",
        options=get_indicators(),
        index=0,
        help="Choose an indicator to analyze"
    )
    
    # Default thresholds based on indicator type
    default_thresholds = {
        'Human Development Index (HDI)': 0.7,  # High HDI threshold
        'GDP per Capita (USD, inflation-adjusted)': 10000.0,
        'Life Expectancy (years)': 70.0,
        'Fertility Rate (children per woman)': 2.1,  # Replacement rate
        'College Completion (Ages 20-24)': 50.0,  # 50% completion rate
        'Higher Secondary Completion (Ages 20-24)': 70.0,
        'Lower Secondary Completion (Ages 20-24)': 80.0,
        'Primary Education Completion (Ages 20-24)': 90.0
    }
    
    # Default threshold based on selected indicator
    default_threshold = next(
        (threshold for name, threshold in default_thresholds.items() 
         if name in selected_indicator),
        50.0  # Default fallback
    )
    
    # Determine if the indicator is a 'good' or 'bad' metric
    is_good_metric = 'Fertility' not in selected_indicator  # Fertility is typically 'bad' when high
    
    # Get min and max values for the selected indicator
    try:
        df = analyzer.load_indicator_data(selected_indicator)
        min_val = float(df['indicator'].min())
        max_val = float(df['indicator'].max())
        
        # Set reasonable step size based on value range
        value_range = max_val - min_val
        step = max(0.1, value_range / 100)  # At most 100 steps
        
        threshold = st.slider(
            "Threshold value",
            min_value=min_val,
            max_value=max_val,
            value=min(max_val, max(min_val, default_threshold)),
            step=step,
            help="Set the threshold value to analyze"
        )
        
        indicator_type = st.radio(
            "Indicator Type",
            ["good", "bad"],
            index=0 if is_good_metric else 1,
            format_func=lambda x: f"Higher is better" if x == "good" else f"Lower is better",
            help="Does a higher value for this indicator represent a better outcome?"
        )
        
        # Year range
        st.subheader("Time Range")
        years = sorted(df['year'].unique())
        if len(years) > 0:  # Added additional check to make sure years is not empty
            if len(years) > 1:
                min_year, max_year = st.slider(
                    "Select year range",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years)))
                )
            else:
                min_year = max_year = int(years[0])
                st.warning(f"Only data for year {min_year} available")
        else:
            st.error("No year data available for the selected indicator.")
            min_year, max_year = 1990, 2020  # Default fallback values
            st.warning(f"Using default year range: {min_year} - {max_year}")
        
    except Exception as e:
        st.error(f"Error loading indicator data: {str(e)}")
        st.stop()

# Main interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Threshold Analysis", 
    "Time Series", 
    "Data Explorer",
    "Development Timeline",
    "Country Comparison"
])

with tab1:
    st.header(f"Threshold Analysis: {selected_indicator}")
    
    # Show indicator description and metadata
    with st.expander("ℹ️ Indicator Information"):
        st.markdown(f"""
        - **Indicator**: {selected_indicator}
        - **Threshold**: {threshold:.2f} ({"higher is better" if indicator_type == "good" else "lower is better"})
        - **Time Period**: {min_year} - {max_year}
        """)
    
    # Process and display results
    with st.spinner("Analyzing data..."):
        try:
            # Get threshold crossing data
            df = analyzer.find_threshold_year(
                indicator_name=selected_indicator,
                threshold=threshold,
                indicator_type=indicator_type,
                min_year=min_year,
                max_year=max_year
            )
            
            if df.empty:
                st.warning("No countries reached the specified threshold in the selected time period.")
            else:
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Countries Reached", len(df))
                with col2:
                    avg_year = df['Year'].mean()
                    st.metric("Average Year", f"{avg_year:.1f}")
                with col3:
                    total_countries = len(analyzer.load_indicator_data(selected_indicator)['country'].unique())
                    pct_total = (len(df) / total_countries) * 100 if total_countries > 0 else 0
                    st.metric("% of Countries", f"{pct_total:.1f}%")
                
                # Show the data table
                st.subheader("Countries That Reached the Threshold")
                st.dataframe(
                    df.sort_values('Year'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Country': st.column_config.TextColumn("Country"),
                        'Year': st.column_config.NumberColumn("Year", format="%d"),
                        'Value': st.column_config.NumberColumn("Value", format="%.2f"),
                        'Threshold': st.column_config.NumberColumn("Threshold", format="%.2f")
                    }
                )
                
                # Show a histogram of years
                st.subheader("Distribution of Threshold Reaching Years")
                fig = px.histogram(
                    df, 
                    x='Year',
                    nbins=20,
                    title=f"When Countries Reached {selected_indicator} ≥ {threshold}",
                    labels={'Year': 'Year', 'count': 'Number of Countries'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing {selected_indicator}: {str(e)}")
    
with tab2:
    st.header(f"Time Series: {selected_indicator}")
    
    try:
        # Get all data for the selected indicator
        df = analyzer.load_indicator_data(selected_indicator)
        
        # Let user select countries to compare
        countries = sorted(df['country'].unique())
        selected_countries = st.multiselect(
            "Select countries to compare",
            options=countries,
            default=countries[:min(5, len(countries))],
            help="Select up to 10 countries to compare",
            key="tab1_country_selector"
        )
        
        if not selected_countries:
            st.info("Please select at least one country to display the time series.")
        else:
            # Filter data for selected countries
            filtered_df = df[df['country'].isin(selected_countries)]
            
            # Create time series plot
            fig = px.line(
                filtered_df,
                x='year',
                y='indicator',
                color='country',
                title=f"{selected_indicator} Over Time",
                labels={'year': 'Year', 'indicator': selected_indicator, 'country': 'Country'}
            )
            
            # Add threshold line if applicable
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.2f}",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Data Table")
            st.dataframe(
                filtered_df.pivot(index='year', columns='country', values='indicator'),
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error loading time series data: {str(e)}")

with tab3:
    st.header("Data Explorer")
    
    try:
        # Get all data for the selected indicator
        df = analyzer.load_indicator_data(selected_indicator)
        
        if not df.empty:
            # Year slider for the map
            year = st.slider(
                "Select year",
                min_value=int(df['year'].min()),
                max_value=int(df['year'].max()),
                value=int(df['year'].max()),
                key="year_slider"
            )
            
            # Filter data for selected year
            year_df = df[df['year'] == year].copy()
            
            # Show world map
            st.subheader(f"World Map - {selected_indicator} ({year})")
            
            # Add ISO-3 country codes for mapping
            try:
                # Try to get country codes from the dataset if available
                if 'iso_alpha3' not in year_df.columns:
                    import pycountry
                    
                    def get_iso_code(country_name):
                        try:
                            return pycountry.countries.get(name=country_name).alpha_3
                        except:
                            try:
                                return pycountry.countries.lookup(country_name).alpha_3
                            except:
                                return None
                    
                    year_df['iso_alpha3'] = year_df['country'].apply(get_iso_code)
                
                # Filter out rows with missing country codes
                map_df = year_df.dropna(subset=['iso_alpha3'])
                
                if not map_df.empty:
                    # Create choropleth map with ISO-3 codes
                    fig = px.choropleth(
                        map_df,
                        locations="iso_alpha3",
                        color="indicator",
                        hover_name="country",
                        hover_data={"iso_alpha3": False, 'indicator': ':.2f'},
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title=f"{selected_indicator} by Country ({year})",
                        labels={'indicator': selected_indicator},
                        projection="natural earth"
                    )
                    
                    # Update layout for better appearance
                    fig.update_layout(
                        coloraxis_colorbar=dict(
                            title=selected_indicator,
                            thicknessmode="pixels", 
                            thickness=20,
                            lenmode="pixels", 
                            len=300,
                            yanchor="top", 
                            y=1,
                            dtick=5
                        ),
                        geo=dict(
                            showframe=False,
                            showcoastlines=True,
                            projection_type='equirectangular'
                        ),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                else:
                    st.warning("No valid country codes found for mapping. Displaying data in a table instead.")
                    fig = None
                    
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.warning("Displaying data in a table instead.")
                fig = None
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Data Table")
            st.dataframe(
                year_df[['country', 'year', 'indicator']].rename(
                    columns={'country': 'Country', 'year': 'Year', 'indicator': selected_indicator}
                ).sort_values('Country'),
                use_container_width=True,
                hide_index=True
            )
            
            # Add export button for the map data
            st.download_button(
                "Download Map Data",
                year_df.to_csv(index=False),
                f"{selected_indicator.replace(' ', '_').lower()}_{year}_map_data.csv",
                "text/csv",
                key="download-map-data"
            )
        else:
            st.warning("No data available for the selected indicator.")
            
    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        if 'df' in locals() and not df.empty:
            st.download_button(
                "Download Available Data",
                df.to_csv(index=False),
                f"{selected_indicator.replace(' ', '_').lower()}_all_data.csv",
                "text/csv",
                key="download-all-data"
            )

# Development Timeline Tab
with tab4:
    st.header("Development Timeline")
    st.markdown("Track how many countries reached the development threshold over time.")
    
    try:
        # Get timeline data
        timeline_df = analyzer.get_development_timeline(
            indicator_name=selected_indicator,
            threshold=threshold,
            indicator_type=indicator_type,
            start_year=min_year,
            end_year=max_year
        )
        
        if timeline_df.empty:
            st.warning("No timeline data available for the selected criteria.")
        else:
            # Plot development timeline
            fig = px.line(
                timeline_df,
                x='Year',
                y='Developed Countries',
                title=f"Development Timeline: {selected_indicator} ≥ {threshold}",
                labels={'Developed Countries': 'Number of Developed Countries'}
            )
            
            # Add total countries line
            fig.add_scatter(
                x=timeline_df['Year'],
                y=timeline_df['Total Countries'],
                mode='lines',
                line=dict(dash='dash'),
                name='Total Countries',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Development Progress")
            st.dataframe(
                timeline_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Year': st.column_config.NumberColumn("Year", format="%d"),
                    'Developed Countries': st.column_config.NumberColumn("Developed Countries", format="%d"),
                    'Total Countries': st.column_config.NumberColumn("Total Countries", format="%d")
                }
            )
            
            # Export button
            csv = timeline_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Timeline Data",
                csv,
                f"{selected_indicator.replace(' ', '_').lower()}_development_timeline.csv",
                "text/csv",
                key="download-timeline"
            )
            
    except Exception as e:
        st.error(f"Error generating development timeline: {str(e)}")

# Country Comparison Tab
with tab5:
    st.header("Country Comparison")
    st.markdown("Compare multiple indicators across different countries and time periods.")
    
    try:
        # Get available countries and indicators
        df = analyzer.load_indicator_data(selected_indicator)
        countries = sorted(df['country'].unique())
        all_indicators = list(analyzer.indicators.keys())
        
        # Multi-select for indicators
        selected_indicators = st.multiselect(
            "Select indicators to compare",
            options=all_indicators,
            default=[selected_indicator],
            help="Select one or more indicators to compare",
            key="tab5_indicator_selector"
        )
        
        # Country selector
        selected_countries = st.multiselect(
            "Select countries to compare",
            options=countries,
            default=countries[:min(3, len(countries))],
            help="Select one or more countries to compare",
            key="tab5_country_selector"
        )
        
        # Year range selector
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        
        year_range = st.slider(
            "Select year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            help="Drag to select a range of years"
        )
        
        if not selected_countries or not selected_indicators:
            st.info("Please select at least one country and one indicator to display the comparison.")
        else:
            # Get data for selected countries, indicators, and years
            comparison_data = []
            
            # If only one indicator is selected, show time series comparison
            if len(selected_indicators) == 1:
                indicator = selected_indicators[0]
                for country in selected_countries:
                    country_data = analyzer.get_country_indicator_history(country, indicator)
                    country_data = country_data[
                        (country_data['Year'] >= year_range[0]) & 
                        (country_data['Year'] <= year_range[1])
                    ]
                    country_data['Country'] = country
                    comparison_data.append(country_data)
                
                if comparison_data:
                    combined_df = pd.concat(comparison_data)
                    
                    # Create line chart
                    fig = px.line(
                        combined_df, 
                        x='Year', 
                        y=indicator,
                        color='Country',
                        title=f"{indicator} Comparison ({year_range[0]}-{year_range[1]})",
                        labels={'value': 'Value', 'variable': 'Country'}
                    )
                    
                    # Add threshold line if available
                    if 'threshold' in locals() and indicator == selected_indicator:
                        fig.add_hline(
                            y=threshold, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Threshold: {threshold}",
                            annotation_position="bottom right"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Comparison Data")
                    st.dataframe(
                        combined_df.pivot(index='Year', columns='Country', values=indicator),
                        use_container_width=True
                    )
            else:
                # For multiple indicators, create a faceted plot
                combined_data = []
                for indicator in selected_indicators:
                    for country in selected_countries:
                        try:
                            country_data = analyzer.get_country_indicator_history(country, indicator)
                            if not country_data.empty:
                                country_data = country_data[
                                    (country_data['Year'] >= year_range[0]) & 
                                    (country_data['Year'] <= year_range[1])
                                ]
                                if not country_data.empty:
                                    # Rename the indicator column to a standard name for melting
                                    indicator_value_col = [col for col in country_data.columns if col not in ['Year', 'country']][0]
                                    country_data = country_data.rename(columns={indicator_value_col: 'Value'})
                                    country_data['Country'] = country
                                    country_data['Indicator'] = indicator
                                    combined_data.append(country_data[['Year', 'Value', 'Country', 'Indicator']])
                        except Exception as e:
                            st.warning(f"Could not load data for {indicator} in {country}: {str(e)}")
                
                if combined_data:
                    combined_df = pd.concat(combined_data)
                    
                    # Create faceted line chart with standardized column names
                    fig = px.line(
                        combined_df, 
                        x='Year', 
                        y='Value',
                        color='Country',
                        facet_col='Indicator',
                        facet_col_wrap=min(2, len(selected_indicators)),
                        title=f"Indicator Comparison ({year_range[0]}-{year_range[1]})",
                        labels={'Value': 'Value'},
                        facet_col_spacing=0.1,
                        height=300 * ((len(selected_indicators) + 1) // 2)
                    )
                    
                    # Update y-axes to be independent and improve layout
                    fig.update_yaxes(matches=None, showticklabels=True)
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    fig.update_layout(
                        margin=dict(l=50, r=50, t=80, b=50),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table with proper formatting
                    st.subheader("Comparison Data")
                    pivot_df = combined_df.pivot_table(
                        index=['Year', 'Country'],
                        columns='Indicator',
                        values='Value'
                    ).reset_index()
                    st.dataframe(pivot_df, use_container_width=True)
            
            if not comparison_data:
                st.warning("No data available for the selected criteria.")
            else:
                comparison_df = pd.concat(comparison_data)
                
                # Plot comparison
                fig = px.line(
                    comparison_df,
                    x='Year',
                    y=selected_indicator,
                    color='Country',
                    title=f"{selected_indicator} Comparison",
                    labels={'Year': 'Year', selected_indicator: selected_indicator}
                )
                
                # Add threshold line if applicable
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.2f}",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.subheader("Comparison Data")
                pivot_df = comparison_df.pivot(
                    index='Year', 
                    columns='Country', 
                    values=selected_indicator
                )
                st.dataframe(pivot_df, use_container_width=True)
                
                # Export button
                csv = comparison_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Comparison Data",
                    csv,
                    f"{selected_indicator.replace(' ', '_').lower()}_comparison.csv",
                    "text/csv",
                    key="download-comparison"
                )
                
    except Exception as e:
        st.error(f"Error generating comparison: {str(e)}")

# Add some documentation
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    This tool helps you analyze when countries reach specific development thresholds.
    
    ### Steps:
    1. **Set your threshold** - The target value to analyze
    2. **Choose indicator type**:
       - *Good*: Higher values are better (e.g., education rates)
       - *Bad*: Lower values are better (e.g., child mortality)
    3. **Select year range** - The period to analyze
    4. **Choose indicators** - Select one or more indicators to analyze
    5. **Click 'Analyze'** - View and export the results
    
    ### Understanding the Results:
    - The table shows when each country first reached the threshold
    - Countries that never reached the threshold are excluded
    - You can download the results as CSV for further analysis
    """)

    # Optionally: search/filter by country
    country = st.text_input("Search for a country:")
    if 'df' in locals() and country:
        filtered = df[df['Country'].str.contains(country, case=False, na=False)]
        st.dataframe(filtered)
