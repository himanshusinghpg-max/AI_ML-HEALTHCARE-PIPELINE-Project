"""
health_ai_pipeline
Complete pipeline:
 synthetic dataset generation 
 preprocessing (imputation, scaling, encoding)
 PCA (visualization, explained variance)
K-Means clustering (elbow + clustering)
 Linear Regression (predict Health Risk Score)
 KNN classification (predict disease presence)
 evaluation and plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
# Synthetic dataset creation
def create_synthetic_medical_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
# Core demographics / vitals
    age = np.random.randint(18, 90, size=n_samples)
    bmi = np.round(np.random.normal(loc=27, scale=5, size=n_samples), 1)  # mean BMI 27
    systolic_bp = np.round(np.random.normal(loc=125, scale=15, size=n_samples))
    diastolic_bp = np.round(np.random.normal(loc=80, scale=10, size=n_samples))
    cholesterol = np.round(np.random.normal(loc=200, scale=40, size=n_samples))
    glucose = np.round(np.random.normal(loc=110, scale=30, size=n_samples))
    heart_rate = np.round(np.random.normal(loc=75, scale=12, size=n_samples))
    smoking = np.random.binomial(1, 0.22, size=n_samples)  #22% smokers
    physical_activity = np.random.randint(0, 8, size=n_samples)  #hours per week
    comorbidity_count = np.random.poisson(0.6, size=n_samples)
    # Composite features and target
    # Higher age, bmi, bp, cholesterol, glucose, smoking, comorbidity = higher health risk
    base_risk = (
        0.03*age +
        0.8*(bmi - 18)/10 +
        0.015*(systolic_bp - 110) +
        0.01*(cholesterol - 150) +
        0.02*(glucose - 90) +
        2*smoking -
        0.2*physical_activity +
        1.5*comorbidity_count
    )
    # normalizing health risk score (higher = worse)
    min_r = base_risk.min()
    max_r = base_risk.max()
    health_score = 100 * (base_risk - min_r) / (max_r - min_r)
    health_score = np.clip(health_score + np.random.normal(0, 5, size=n_samples), 0, 100).round(1)
    # Simulate disease presence 
    # Probability increases with health_score
    disease_prob = 1 / (1 + np.exp(-(health_score - 40)/10))  # sigmoid
    disease = np.random.binomial(1, disease_prob)
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'heart_rate': heart_rate,
        'smoking': smoking,
        'physical_activity': physical_activity,
        'comorbidity_count': comorbidity_count,
        'health_score': health_score,
        'disease': disease
    })

    # Introduce some missing values (simulating healthcare data)
    for col in ['bmi', 'cholesterol', 'glucose']:
        mask = np.random.rand(n_samples) < 0.03  # 3% NaNs
        df.loc[mask, col] = np.nan

    return shuffle(df, random_state=random_state)
#  Preprocessing
def preprocess(df, numeric_features):
    # Impute missing numeric features with median
    imputer = SimpleImputer(strategy='median')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])

    # Separate features & targets
    X = df.drop(columns=['health_score', 'disease'])
    y_reg = df['health_score']         # regression target
    y_clf = df['disease']             # classification target
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y_reg, y_clf, scaler
# PCA - dimensionality reduction & visualization
def run_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    return pca, X_pca, explained
# K-Means Clustering + Elbow method
def find_k_elbow(X_scaled, k_range=(1, 10)):
    inertia = []
    Ks = list(range(k_range[0], k_range[1] + 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return Ks, inertia
def cluster_and_label(X_scaled, n_clusters=3):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    return km, labels
# Regression - Linear Regression for health_score
def regression_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, preds),
        'MSE': mean_squared_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }
    return model, preds, metrics
# 6. KNN classifier for disease prediction
def knn_classification(X_train, y_train, X_test, y_test, param_search=True):
    if param_search:
        knn = KNeighborsClassifier()
        grid = {'n_neighbors': [3,5,7,9], 'weights': ['uniform', 'distance']}
        gs = GridSearchCV(knn, grid, cv=5, scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        best_params = None
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'classification_report': classification_report(y_test, preds, digits=3),
        'best_params': best_params
    }
    return model, preds, probs, metrics
# Utility: plotting functions
def plot_elbow(Ks, inertia):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, inertia, marker='o')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_pca_clusters(X_pca, labels, title='PCA projection with clusters'):
    plt.figure(figsize=(7,5))
    palette = sns.color_palette("tab10", np.unique(labels).max()+1)
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette=palette, s=40, alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
def plot_regression_results(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([0,100],[0,100], '--', color='grey')
    plt.xlabel('Actual Health Score')
    plt.ylabel('Predicted Health Score')
    plt.title('Actual vs Predicted Health Score')
    plt.tight_layout()
    plt.show()
def plot_confusion(cm, labels=[0,1]):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
# Main pipeline
def main():
    # Generating data
    df = create_synthetic_medical_data(n_samples=2000)
    print("Dataset sample:")
    print(df.head())
    numeric_features = ['age','bmi','systolic_bp','diastolic_bp','cholesterol','glucose',
                        'heart_rate','smoking','physical_activity','comorbidity_count']

# Preprocess
    X_df, X_scaled, y_reg, y_clf, scaler = preprocess(df, numeric_features)
    print(f"\nFeatures scaled shape: {X_scaled.shape}")
# PCA for visualization & reduce purchase-like features dimension
    pca, X_pca2, explained = run_pca(X_scaled, n_components=2)
    print(f"\nPCA explained variance ratio (2 components): {explained}")
# Elbow method find optimal k
    Ks, inertia = find_k_elbow(X_scaled, k_range=(1,10))
    plot_elbow(Ks, inertia)
# Choose k = 3 (Low/Medium/High risk) - we can base this on elbow plot
    km, labels = cluster_and_label(X_scaled, n_clusters=3)
    df['cluster'] = labels
    # Visualize clusters on PCA projection
    plot_pca_clusters(X_pca2, labels, title='Patient clusters (PCA projection)')
    # Examine cluster centers (in original scaled space -> inverse transform for interpretation)
    centers_scaled = km.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_unscaled, columns=X_df.columns)
    centers_df.index.name = 'cluster'
    print("\nCluster centers (approx in original feature scale):")
    print(centers_df.round(2))
    # Regression: predict health_score
    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
    reg_model, reg_preds, reg_metrics = regression_model(X_train, y_train_reg, X_test, y_test_reg)
    print("\nRegression metrics for Health Score prediction:")
    for k,v in reg_metrics.items():
        print(f" {k}: {v:.4f}")

    plot_regression_results(y_test_reg, reg_preds)
    # KNN classification: predict disease
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
    knn_model, knn_preds, knn_probs, knn_metrics = knn_classification(X_train_c, y_train_c, X_test_c, y_test_c, param_search=True)
    print("\nKNN classification metrics:")
    print(f" Accuracy: {knn_metrics['accuracy']:.4f}")
    print(" Classification Report:\n", knn_metrics['classification_report'])
    print(" Best params (GridSearch):", knn_metrics['best_params'])
    plot_confusion(knn_metrics['confusion_matrix'], labels=[0,1])

    #Integrate results: build a small summary table for a few patients
    # Use PCA coordinates, cluster label, predicted health score, predicted disease probability
    demo_idx = np.random.choice(len(df), size=8, replace=False)
    demo_df = df.iloc[demo_idx].copy().reset_index(drop=True)
    demo_scaled = scaler.transform(demo_df.drop(columns=['health_score','disease','cluster']))
    demo_pca = pca.transform(demo_scaled)
    demo_clusters = km.predict(demo_scaled)
    demo_health_pred = reg_model.predict(demo_scaled).round(1)
    demo_disease_prob = knn_model.predict_proba(demo_scaled)[:,1] if hasattr(knn_model, "predict_proba") else knn_model.predict(demo_scaled)
    demo_df['pred_health_score'] = demo_health_pred
    demo_df['pred_disease_prob'] = np.round(demo_disease_prob, 3)
    demo_df['pca1'] = np.round(demo_pca[:,0], 3)
    demo_df['pca2'] = np.round(demo_pca[:,1], 3)
    demo_df['pred_cluster'] = demo_clusters
    print("\nDemo patient summary (predictions + clusters):")
    print(demo_df[['age','bmi','systolic_bp','cholesterol','health_score','pred_health_score','pred_disease_prob','pred_cluster','pca1','pca2']])
    # Saved models outputs as needed
if __name__ == "__main__":
    main()
