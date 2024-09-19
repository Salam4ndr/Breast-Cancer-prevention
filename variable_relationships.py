import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_relationships(data):
    """
    Analyze relationships between variables in the dataset by visualizing correlations and pairwise relationships.
    
    Parameters:
    - data (DataFrame): Pandas DataFrame containing the dataset.
    
    Returns:
    - DataFrame: The cleaned DataFrame after removing multicollinear variables.
    """
    # Plot the relationship between mean radius and mean area
    plt.figure(figsize=(15, 10))
    sns.jointplot(x='radius_mean', y='area_mean', data=data, kind='reg')
    plt.title('Relationship between Mean Radius and Mean Area')
    plt.show()
    
    # Plot pairwise relationships for selected columns
    cols = ['diagnosis',
            'radius_mean',
            'texture_mean',
            'perimeter_mean',
            'area_mean',
            'smoothness_mean',
            'compactness_mean',
            'concavity_mean',
            'concave points_mean',
            'symmetry_mean',
            'fractal_dimension_mean']
    
    sns.pairplot(data=data[cols], hue='diagnosis', palette='rocket')
    plt.title('Pairwise Relationships')
    plt.show()
    
    # Plot the correlation matrix
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".2f", ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation Map')
    plt.show()
    
    # Drop highly correlated features to address multicollinearity
    cols_to_drop = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                    'smoothness_worst', 'compactness_worst', 'concavity_worst',
                    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
                    'perimeter_mean', 'perimeter_se', 'area_mean', 'area_se',
                    'concavity_mean', 'concavity_se', 'concave points_mean', 'concave points_se']
    
    data = data.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Plot the updated correlation matrix
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".2f", ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Updated Correlation Map')
    plt.show()
    
    return data