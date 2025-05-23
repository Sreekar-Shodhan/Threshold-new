import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import os
import yaml
from pathlib import Path

@dataclass
class IndicatorTriplet:
    """
    Represents a single data point with country, year, and indicator value.
    """
    country: str
    year: int
    indicator: float

class IndicatorAnalyzer:
    def __init__(self, data: List[Dict[str, Union[str, int, float]]] = None, data_dir: str = 'DATACSV'):
        """
        Initialize the analyzer with optional initial data.
        
        Args:
            data: List of dictionaries with 'country', 'year', and 'indicator' keys
            data_dir: Directory containing indicator data files (default: 'DATACSV')
        """
        self.triplets = []
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        self.indicators = {}
        self._load_available_indicators()
        
        if data:
            self.add_triplets(data)
    
    def add_triplet(self, country: str, year: int, indicator: float) -> None:
        """Add a single triplet to the analyzer."""
        self.triplets.append(IndicatorTriplet(country, year, indicator))
    
    def add_triplets(self, data: List[Dict[str, Union[str, int, float]]]) -> None:
        """
        Add multiple triplets to the analyzer.
        
        Args:
            data: List of dictionaries with 'country', 'year', and 'indicator' keys
        """
        for item in data:
            self.add_triplet(
                country=str(item['country']),
                year=int(item['year']),
                indicator=float(item['indicator'])
            )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the triplets to a pandas DataFrame."""
        return pd.DataFrame([
            {'country': t.country, 'year': t.year, 'indicator': t.indicator}
            for t in self.triplets
        ])
    
    def get_countries(self) -> List[str]:
        """Get a list of unique countries in the dataset."""
        return sorted(list(set(t.country for t in self.triplets)))
    
    def get_years(self) -> List[int]:
        """Get a list of unique years in the dataset."""
        return sorted(list(set(t.year for t in self.triplets)))
    
    def filter_by_country(self, country: str) -> List[IndicatorTriplet]:
        """Filter triplets by country."""
        return [t for t in self.triplets if t.country == country]
    
    def filter_by_year(self, year: int) -> List[IndicatorTriplet]:
        """Filter triplets by year."""
        return [t for t in self.triplets if t.year == year]
    
    def get_country_time_series(self, country: str) -> Dict[int, float]:
        """
        Get a time series of indicator values for a specific country.
        
        Returns:
            Dictionary mapping years to indicator values
        """
        return {
            t.year: t.indicator 
            for t in self.filter_by_country(country)
        }
    
    def get_yearly_average(self, year: int) -> float:
        """Calculate the average indicator value for a specific year across all countries."""
        year_data = self.filter_by_year(year)
        if not year_data:
            return None
        return sum(t.indicator for t in year_data) / len(year_data)
    
    def get_country_average(self, country: str) -> float:
        """Calculate the average indicator value for a specific country across all years."""
        country_data = self.filter_by_country(country)
        if not country_data:
            return None
        return sum(t.indicator for t in country_data) / len(country_data)
    
    def get_yearly_summary(self) -> Dict[int, dict]:
        """
        Generate summary statistics for each year.
        
        Returns:
            Dictionary mapping years to statistics (min, max, avg, count)
        """
        years = self.get_years()
        summary = {}
        for year in years:
            indicators = [t.indicator for t in self.filter_by_year(year)]
            if not indicators:
                continue
            summary[year] = {
                'min': min(indicators),
                'max': max(indicators),
                'avg': sum(indicators) / len(indicators),
                'count': len(indicators)
            }
        return summary
        
    def _load_available_indicators(self) -> None:
        """Load available indicators from the data directory."""
        # Map of file names to display names
        display_names = {
            'human-development-index': 'Human Development Index (HDI)',
            'gdppercapita_us_inflation_adjusted': 'GDP per Capita (USD, inflation-adjusted)',
            'life_expectancy_years': 'Life Expectancy (years)',
            'children_per_woman_total_fertility': 'Fertility Rate (children per woman)',
            '20-24-College_comp': 'College Completion (Ages 20-24)',
            '20-24-Higher_Secondary_fin': 'Higher Secondary Completion (Ages 20-24)',
            '20-24-Lower_Secondary_fin': 'Lower Secondary Completion (Ages 20-24)',
            '20-24-Primary_fin': 'Primary Education Completion (Ages 20-24)'
        }
        
        for file in self.data_dir.glob('*.csv'):
            indicator_name = file.stem
            display_name = display_names.get(indicator_name, indicator_name.replace('_', ' ').title())
            self.indicators[display_name] = str(file)
    
    def load_indicator_data(self, indicator_name: str) -> pd.DataFrame:
        """
        Load data for a specific indicator.
        
        Args:
            indicator_name: Display name of the indicator to load
            
        Returns:
            DataFrame with the indicator data in long format (country, year, value)
        """
        try:
            if indicator_name not in self.indicators:
                raise ValueError(f"Indicator '{indicator_name}' not found in available indicators")
                
            file_path = self.indicators[indicator_name]
            print(f"Loading data from: {file_path}")
            
            # Read the CSV file with additional error handling
            try:
                df = pd.read_csv(file_path)
                print(f"CSV loaded successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
                # Return empty DataFrame with proper structure
                return pd.DataFrame(columns=['country', 'year', 'indicator'])
            
            # Reset triplets when loading new data
            self.triplets = []
            
            # Handle common file formats - first check OWID standard format
            if 'Entity' in df.columns and 'Year' in df.columns:
                print("Detected OWID format")
                value_col = next((col for col in df.columns if col not in ['Entity', 'Year', 'Code']), 'Value')
                df = df.rename(columns={
                    'Entity': 'country',
                    'Year': 'year',
                    value_col: 'indicator'
                })
            # Check for standard column names
            elif 'country' in df.columns and 'year' in df.columns:
                print("Detected standard format")
                value_col = next((col for col in df.columns if col not in ['country', 'year', 'code']), 'value')
                df = df.rename(columns={value_col: 'indicator'})
            # Check for wide format (country in first column, years as other columns)
            elif len(df.columns) > 2:
                print("Attempting to convert from wide format")
                country_col = df.columns[0]
                # Melt the dataframe to convert from wide to long format
                try:
                    df = df.melt(
                        id_vars=[country_col],
                        var_name='year',
                        value_name='indicator'
                    ).rename(columns={country_col: 'country'})
                    print(f"Melted to long format. New shape: {df.shape}")
                except Exception as e:
                    print(f"Error melting DataFrame: {str(e)}")
                    # Provide a fallback empty DataFrame
                    return pd.DataFrame(columns=['country', 'year', 'indicator'])
            
            # Ensure required columns exist
            missing_cols = [col for col in ['country', 'year', 'indicator'] if col not in df.columns]
            if missing_cols:
                print(f"Missing columns: {missing_cols}")
                # Try to infer missing columns if possible
                if 'country' not in df.columns and df.shape[1] > 0:
                    df['country'] = df.iloc[:, 0] if df.shape[1] > 0 else 'Unknown'
                if 'year' not in df.columns and df.shape[1] > 1:
                    df['year'] = df.iloc[:, 1] if df.shape[1] > 1 else 2020
                if 'indicator' not in df.columns and df.shape[1] > 2:
                    df['indicator'] = df.iloc[:, 2] if df.shape[1] > 2 else 0.0
            
            # Clean up year column - more robust parsing
            if 'year' in df.columns:
                print(f"Year column dtype before cleaning: {df['year'].dtype}")
                if df['year'].dtype != 'int64':
                    # Try to extract years from strings like '1990' or '1990s' or 'year_1990'
                    try:
                        df['year'] = pd.to_numeric(df['year'].astype(str).str.extract('(\d{4}|\d{2})', expand=False), errors='coerce')
                        # Fix 2-digit years
                        mask_2digit = (df['year'] < 100) & (df['year'] >= 0)
                        df.loc[mask_2digit, 'year'] = df.loc[mask_2digit, 'year'] + 1900
                        # Drop rows with missing years
                        df = df.dropna(subset=['year'])
                        df['year'] = df['year'].astype(int)
                        print(f"Cleaned years, count: {len(df['year'].unique())}")
                    except Exception as e:
                        print(f"Error cleaning years: {str(e)}")
            
            # Clean up indicator values
            if 'indicator' in df.columns:
                print(f"Indicator column dtype before cleaning: {df['indicator'].dtype}")
                df['indicator'] = pd.to_numeric(df['indicator'], errors='coerce')
            
            # Print some diagnostics
            print(f"Final DataFrame shape: {df.shape}")
            print(f"Unique countries: {len(df['country'].unique())}")
            print(f"Unique years: {sorted(df['year'].unique().tolist()) if 'year' in df.columns else 'No year column'}")
            
            # Ensure all required columns exist and drop rows with NaN values
            result_df = df.reindex(columns=['country', 'year', 'indicator']).dropna()
            print(f"Final clean data shape: {result_df.shape}")
            
            # Add data to analyzer
            if not result_df.empty:
                self.add_triplets(result_df.to_dict('records'))
            
            return result_df
        except Exception as e:
            print(f"Error in load_indicator_data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return empty DataFrame with proper structure
            return pd.DataFrame(columns=['country', 'year', 'indicator'])
    
    def find_threshold_year(self, 
                          indicator_name: str, 
                          threshold: float, 
                          indicator_type: str = 'good',
                          min_year: int = None,
                          max_year: int = None) -> pd.DataFrame:
        """
        Find the first year each country crosses a threshold.
        
        Args:
            indicator_name: Name of the indicator to analyze
            threshold: Threshold value to check against
            indicator_type: 'good' (higher is better) or 'bad' (lower is better)
            min_year: Minimum year to consider
            max_year: Maximum year to consider
            
        Returns:
            DataFrame with countries and the year they crossed the threshold
        """
        # Load the indicator data
        df = self.load_indicator_data(indicator_name)
        
        # Filter years if specified
        if 'year' in df.columns:
            if min_year is not None:
                df = df[df['year'] >= min_year]
            if max_year is not None:
                df = df[df['year'] <= max_year]
        
        # Find threshold crossing years
        results = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country].sort_values('year')
            
            if indicator_type == 'good':
                # Find first year where value >= threshold
                mask = country_data['indicator'] >= threshold
            else:  # 'bad'
                # Find first year where value <= threshold
                mask = country_data['indicator'] <= threshold
                
            # Add robust error handling to prevent index errors
            try:
                if mask.any() and not country_data[mask].empty:
                    first_year = country_data[mask].iloc[0]['year']
                    value = country_data[mask].iloc[0]['indicator']
                    results.append({
                        'Country': country,
                        'Year': first_year,
                        'Value': value,
                        'Threshold': threshold
                    })
            except (IndexError, KeyError) as e:
                # Skip this country if there's an error
                print(f"Error processing {country}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
        
    def get_development_timeline(self, indicator_name: str, threshold: float, 
                              indicator_type: str = 'good', 
                              start_year: int = 1960, 
                              end_year: int = 2025) -> pd.DataFrame:
        """
        Generate a timeline of how many countries were developed each year.
        
        Args:
            indicator_name: Name of the indicator
            threshold: Development threshold
            indicator_type: 'good' or 'bad'
            start_year: Start year for the timeline
            end_year: End year for the timeline
            
        Returns:
            DataFrame with year and count of developed countries
        """
        try:
            # Get threshold years for all countries
            df = self.find_threshold_year(indicator_name, threshold, indicator_type, start_year, end_year)
            
            # Create a year range
            years = range(start_year, end_year + 1)
            timeline = []
            
            # Add safety check for empty DataFrame
            total_countries = len(df['Country'].unique()) if not df.empty else 0
            
            # Count how many countries were developed by each year
            for year in years:
                count = len(df[df['Year'] <= year]) if not df.empty else 0
                timeline.append({
                    'Year': year,
                    'Developed Countries': count,
                    'Total Countries': total_countries
                })
                
            return pd.DataFrame(timeline)
        except Exception as e:
            print(f"Error in get_development_timeline: {str(e)}")
            # Return an empty timeline with the requested years
            timeline = [{'Year': year, 'Developed Countries': 0, 'Total Countries': 0} for year in range(start_year, end_year + 1)]
            return pd.DataFrame(timeline)
    
    def get_country_indicator_history(self, country: str, indicator_name: str) -> pd.DataFrame:
        """
        Get the complete history of an indicator for a specific country.
        
        Args:
            country: Country name
            indicator_name: Name of the indicator
            
        Returns:
            DataFrame with year and indicator value
        """
        df = self.load_indicator_data(indicator_name)
        country_data = df[df['country'] == country].sort_values('year')
        return country_data[['year', 'indicator']].rename(
            columns={'year': 'Year', 'indicator': indicator_name}
        )
    
    def compare_indicators(self, indicators: list, country: str = None) -> pd.DataFrame:
        """
        Compare multiple indicators for a specific country or all countries.
        
        Args:
            indicators: List of indicator names
            country: Optional country name to filter by
            
        Returns:
            DataFrame with combined indicator data
        """
        result_dfs = []
        
        for indicator in indicators:
            df = self.load_indicator_data(indicator)
            if country:
                df = df[df['country'] == country]
            df = df.rename(columns={'indicator': indicator})
            result_dfs.append(df)
        
        # Merge all indicator dataframes
        if not result_dfs:
            return pd.DataFrame()
            
        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = pd.merge(result, df, on=['country', 'year'], how='outer')
            
        return result

# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = [
        {"country": "USA", "year": 2020, "indicator": 1.5},
        {"country": "USA", "year": 2021, "indicator": 1.7},
        {"country": "Canada", "year": 2020, "indicator": 1.2},
        {"country": "Canada", "year": 2021, "indicator": 1.3},
    ]
    
    # Create analyzer and add data
    analyzer = IndicatorAnalyzer(sample_data)
    
    # Print some statistics
    print(f"Countries: {analyzer.get_countries()}")
    print(f"Years: {analyzer.get_years()}")
    print(f"USA time series: {analyzer.get_country_time_series('USA')}")
    print(f"2020 average: {analyzer.get_yearly_average(2020):.2f}")
    print("Yearly summary:", analyzer.get_yearly_summary())