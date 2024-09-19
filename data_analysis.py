import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def perform_data_analysis(data):
    """
    Perform data analysis and visualization on the given dataset.
    
    Args:
    data (pd.DataFrame): The dataset to analyze.
    """
    # Ensure that 'data' is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Error: The provided data is not a DataFrame.")
        return

    # Ensure 'diagnosis' column is present
    if 'diagnosis' not in data.columns:
        print("Error: The 'diagnosis' column is missing from the dataset.")
        return

    try:
        # Plot histogram of diagnosis frequencies
        frequencies = data['diagnosis'].value_counts()
        colors = ['green', 'red']
        plt.bar(frequencies.index, frequencies.values, color=colors)
        plt.xticks(frequencies.index, ['Benign', 'Malignant'])
        plt.xlabel('Diagnosis')
        plt.ylabel('Frequency')
        plt.title('Histogram of Diagnosis')
        plt.show()
    except Exception as e:
        print(f"Error plotting histogram: {e}")

    # List of features for histograms and KDE plots
    features_list = [
        "radius_mean", "texture_mean", "smoothness_mean",
        "compactness_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "smoothness_se", "compactness_se",
        "symmetry_se", "fractal_dimension_se"
    ]

    try:
        # Plot histograms for each feature
        for feature in features_list:
            plt.hist(data[data["diagnosis"] == 1][feature], bins=30, alpha=0.5, color='red', label="Malignant")
            plt.hist(data[data["diagnosis"] == 0][feature], bins=30, alpha=0.5, color='green', label="Benign")
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {feature}')
            plt.legend()
            plt.show()
    except Exception as e:
        print(f"Error plotting histograms: {e}")

    try:
        # Plot KDE for each feature
        for feature in features_list:
            data[data["diagnosis"] == 1][feature].plot.kde(label="Malignant", color="red")
            data[data["diagnosis"] == 0][feature].plot.kde(label="Benign", color="green")
            plt.title(f'KDE of {feature}')
            plt.legend()
            plt.show()
    except Exception as e:
        print(f"Error plotting KDEs: {e}")

    try:
        # Principal Component Analysis (PCA)
        y = data["diagnosis"]
        X = data.drop(["diagnosis"], axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=5)
        pca.fit(X_scaled)
        variance_ratio = pca.explained_variance_ratio_

        plt.bar(range(len(variance_ratio)), variance_ratio)
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        for i, val in enumerate(variance_ratio):
            plt.text(i, val, f'{val*100:.2f}%')
        plt.title('PCA Explained Variance Ratio')
        plt.show()
    except Exception as e:
        print(f"Error performing PCA: {e}")

    try:
        pca = PCA(n_components=3)
        X_reduced_pca = pca.fit_transform(X_scaled)

        pca_data = pd.DataFrame(X_reduced_pca, columns=["p1", "p2", "p3"])
        pca_data["diagnosis"] = y

        data = [
            go.Scatter3d(
                x=pca_data.p1,
                y=pca_data.p2,
                z=pca_data.p3,
                mode='markers',
                marker=dict(
                    size=5,
                    color=pca_data["diagnosis"],
                    colorscale='Viridis',
                    line=dict(width=2)
                )
            )
        ]

        layout = go.Layout(
            title="PCA 3D Plot",
            scene=dict(
                xaxis=dict(title="p1"),
                yaxis=dict(title="p2"),
                zaxis=dict(title="p3")
            ),
            hovermode="closest"
        )

        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename='pca_3d_plot.html')
    except Exception as e:
        print(f"Error creating PCA 3D plot: {e}")

    try:
        pca = PCA(n_components=2)
        X_reduced_pca = pca.fit_transform(X_scaled)

        pca_data = pd.DataFrame(X_reduced_pca, columns=["p1", "p2"])
        pca_data["diagnosis"] = y

        data = [
            go.Scatter(
                x=pca_data.p1,
                y=pca_data.p2,
                mode='markers',
                marker=dict(
                    size=10,
                    color=pca_data["diagnosis"],
                    colorscale='Viridis',
                    line=dict(width=2)
                )
            )
        ]

        layout = go.Layout(
            title="PCA 2D Plot",
            xaxis=dict(title="p1"),
            yaxis=dict(title="p2"),
            hovermode="closest"
        )

        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename='pca_2d_plot.html')
    except Exception as e:
        print(f"Error creating PCA 2D plot: {e}")

    try:
        # Detect outliers using DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X_reduced_pca)

        outliers = pca_data[dbscan.labels_ == -1]

        data = [
            go.Scatter(
                x=pca_data.p1,
                y=pca_data.p2,
                mode='markers',
                marker=dict(
                    size=10,
                    color=pca_data["diagnosis"],
                    colorscale='Viridis',
                    line=dict(width=2)
                )
            ),
            go.Scatter(
                x=outliers.p1,
                y=outliers.p2,
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    symbol='circle',
                    line=dict(width=2)
                ),
                name='Outliers'
            )
        ]

        layout = go.Layout(
            title="PCA 2D Plot with DBSCAN Outliers",
            xaxis=dict(title="p1"),
            yaxis=dict(title="p2"),
            hovermode="closest"
        )

        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename='pca_dbscan_outliers.html')
    except Exception as e:
        print(f"Error performing DBSCAN clustering: {e}")

    try:
        # Boxplots for each feature
        for feature in features_list:
            if feature in data.columns:  # Ensure feature exists in data
                melted_data = pd.melt(data, id_vars="diagnosis", value_vars=[feature])
                plt.figure(figsize=(8, 6))
                sns.boxplot(x="variable", y="value", hue="diagnosis", data=melted_data)
                plt.title(f'Boxplot of {feature}')
                plt.show()
            else:
                print(f"Warning: Feature '{feature}' is not present in the dataset.")
    except Exception as e:
        print(f"")

    print("Data analysis complete.")
