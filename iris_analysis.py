import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
from datetime import datetime, timedelta

def load_and_prepare_data():
    """
    Load the Iris dataset and prepare it for analysis by adding a time series component
    """
    try:
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_name)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Add a synthetic time series component for demostration
        start_date = datetime(2024, 1, 1)
        df['date'] = [start_date + timedelta(days=x) for x in range(len(df))]

        # Add some random missing values for demostration
        df.loc[np.random.choice(df.index, 10), 'sepal length (cm)'] = np.nan

        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
def explore_data(df):
    """
    Perform initial data exploration
    """
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

def clean_data(df):
    """
    Clean the dataset by handling missing values
    """
    # Fill missing values with median of respective column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    return df

def perform_analysis(df):
    """
    Perform basic statistical analysis
    """
    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMean values by species:")
    print(df.groupby('species').mean())

def create_visualizations(df):
    """
    Create and save various visualizations
    """
    # Set the style for all plots
    plt.style.use('seaborn')

    # 1. Line Chart - Time series of sepal length
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['sepal length (cm)'], label='sepal Length')
    plt.title('Sepal Length Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sepal Length (cm)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('line_chart.png')
    plt.close()

    # 2. Bar Chart - Average measurements by species
    plt.figure(figsize=(10, 6))
    species_means = df.groupby('species').mean()
    species_means[['sepal length (cm)', 'sepal width (sm)',
                   'petal length (cm)', 'petal width (cm)'
                   ]].plot(kind='bar')
    plt.title('Average Measurements by Species')
    plt.xlabel('Species')
    plt.ylabel('Centimeters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('barr_chart.png')
    plt.close()

    # 3. Histogram - Distribution of sepal length
    plt.figure(figsize=(10, 6))
    plt.hist(df['sepal length (cm)'], bins=20, edgecolor='black')
    plt.title('Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('histogram.png')
    plt.close()

    # 4. Scatter Plot - Sepal length vs Petal length by species
    plt.figure(figsize=(10, 6))
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'],
                    species_data['petal length (cm)'],
                    label=species, alpha=0.6)
        plt.title('Sepal Length vs Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('scatter_plot.png')
        plt.close()

def main():
    """
    Main function to orchestrate the analysis
    """
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Explore data
    explore_data(df)

    # Clean data
    explore_data(df)

    # Perform analysis
    perform_analysis(df)

    # Create visualizations
    create_visualizations(df)

    print("\nAnalysis complete! All visualizations have been saved.")

if __name__ == "__main__":
    main()