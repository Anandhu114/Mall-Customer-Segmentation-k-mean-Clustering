# Mall Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

Mall Customer Segmentation is an unsupervised machine learning project that groups customers according to similarities in their income, spending behavior, and age.

The project uses the **K-Means Clustering algorithm** to identify customer segments from the `Mall_Customers.csv` dataset. These segments can help businesses understand different customer groups and plan targeted marketing strategies.

The project also includes Exploratory Data Analysis (EDA), data visualization, feature scaling, the Elbow Method, and cluster visualization.

## 🎯 Project Objectives

- Analyze customer demographic and spending data.
- Perform Exploratory Data Analysis (EDA) to understand customer characteristics.
- Visualize age, annual income, gender, and spending score distributions.
- Apply feature scaling using `StandardScaler`.
- Implement K-Means clustering to segment customers.
- Use the Elbow Method to examine the appropriate number of clusters.
- Visualize customer segments using 2D scatter plots and 3D plots.
- Display customer IDs belonging to each cluster.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data loading and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical charts and plots |
| Scikit-learn | Feature scaling and K-Means clustering |
| Jupyter Notebook | Interactive development and analysis |

## 📂 Dataset Description

The project loads a dataset named `Mall_Customers.csv`.

The notebook uses the following columns:

| Column | Description |
|---|---|
| `CustomerID` | Customer identifier |
| `Genre` / `Gender` | Customer gender; the notebook renames `Genre` to `Gender` |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income, expressed in thousands of dollars |
| `Spending Score (1-100)` | Spending score on a scale from 1 to 100 |

**Note:** The CSV file must be available in the working directory when you run the notebook. The dataset file itself is not included in the uploaded notebook.

## 🔍 Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("Mall_Customers.csv")
```

The notebook inspects the dataset shape, data types, descriptive statistics, and missing values.

### 2. Data Preprocessing

The notebook renames the `Genre` column to `Gender` and checks the data for missing values.

```python
df.rename(columns={"Genre": "Gender"}, inplace=True)

df.info()
df.describe()
df.isna().sum()
```

### 3. Exploratory Data Analysis (EDA)

The notebook explores customer characteristics using visualizations, including:

- Box plots for annual income and spending score.
- Distribution plots for age, annual income, and spending score.
- A count plot for gender distribution.
- Scatter plots comparing age, income, and spending score.
- Bar plots showing customer age groups.
- Bar plots intended to explore spending score and income ranges.

These visualizations help examine the data before applying clustering.

### 4. Feature Scaling

The project uses `StandardScaler` to standardize the numerical features.

```python
df_scaled = df[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
]

scaler = StandardScaler()
df_scaled_fit = scaler.fit_transform(df_scaled)
```

Feature scaling is important for distance-based algorithms such as K-Means because variables with larger numerical ranges can otherwise have a greater influence on the clustering results.

### 5. K-Means Clustering

The notebook initially tests K-Means with four clusters, then applies a five-cluster configuration.

The primary clustering analysis uses:

- Annual Income
- Spending Score

```python
var_list = df_scaled_fit[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

kmeans = KMeans(n_clusters=5, max_iter=50)
kmeans.fit(var_list)

df["label"] = kmeans.labels_
```

The resulting cluster labels are added to the customer DataFrame.

**Important:** K-Means cluster labels such as `0`, `1`, `2`, `3`, and `4` are identifiers assigned by the algorithm. They do not represent rankings.

### 6. Finding the Number of Clusters: Elbow Method

The notebook calculates the K-Means inertia for cluster counts from 1 to 10.

```python
ssd = []

for num_clusters in range(1, 11):
    kmeans = KMeans(
        n_clusters=num_clusters,
        max_iter=50
    )
    kmeans.fit(var_list)
    ssd.append(kmeans.inertia_)
```

The resulting values are plotted to create an Elbow Curve.

The Elbow Method helps identify a cluster count where increasing the number of clusters begins to provide smaller reductions in within-cluster sum of squared distances.

The notebook subsequently chooses five clusters for its main analysis. The actual optimal cluster count should be justified by examining the plotted curve and, if needed, additional clustering evaluation methods.

### 7. Visualizing Customer Segments

The project visualizes the five clusters through:

- A 2D scatter plot of annual income versus spending score.
- Box plots comparing income across cluster labels.
- Box plots comparing spending scores across cluster labels.
- A 3D scatter plot using age, annual income, and spending score.

These plots help compare the characteristics of the groups produced by the algorithm.

### 8. Customer Group Analysis

The notebook filters the DataFrame by cluster label and prints the number of customers and their customer IDs.

This provides a way to inspect which customers belong to each generated segment.

## 📊 Main Project Features

- Customer data inspection and preprocessing.
- Exploratory data analysis and statistical visualizations.
- Customer age and income distribution analysis.
- Numerical feature scaling with `StandardScaler`.
- K-Means clustering with different cluster counts.
- Elbow Curve analysis using K-Means inertia.
- Two-dimensional customer segment visualization.
- Three-dimensional clustering visualization.
- Customer counts and IDs for each cluster.

## 🚀 Getting Started

### Prerequisites

Install Python and Jupyter Notebook or JupyterLab.

### 1. Clone the Repository

Replace the example URL below with the URL of your actual GitHub repository.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Add the Dataset

Place `Mall_Customers.csv` in the project directory, alongside the notebook, or update the CSV path in the notebook to match your dataset location.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `mall-customer-segmentation-k-mean-clustering.ipynb` and execute the notebook cells in order.

## 📁 Suggested Repository Structure

```text
mall-customer-segmentation/
│
├── mall-customer-segmentation-k-mean-clustering.ipynb
├── Mall_Customers.csv
├── images/
│   ├── eda-visualizations.png
│   ├── elbow-curve.png
│   ├── customer-clusters.png
│   └── customer-clusters-3d.png
│
├── requirements.txt
└── README.md
```

This is a suggested structure for organizing the GitHub repository, not a verified listing of the current project files.

## 💡 Applications

Customer segmentation can support business activities such as:

- Targeted marketing campaigns.
- Customer behavior analysis.
- Customer group profiling.
- Identifying groups with different spending patterns.
- Supporting customer engagement and promotional planning.

These are potential applications of customer segmentation; the notebook does not implement a complete marketing or recommendation system.

## ⚠️ Limitations and Future Improvements

Possible improvements to the current notebook include:

- Add a fixed `random_state` to make clustering results more reproducible.
- Compare cluster counts using silhouette scores in addition to the Elbow Method.
- Improve the spending-score range visualization; the current notebook mistakenly uses age-group counts for its spending-score chart.
- Update deprecated Seaborn plotting functions such as `distplot`.
- Summarize each cluster using mean age, annual income, and spending score.
- Add a more detailed interpretation of the customer groups.
- Save the fitted scaler and clustering model for reuse.
- Build an interactive dashboard using Streamlit or another suitable framework.

## 👨‍💻 Author

**Anandhu P S**

- GitHub: https://github.com/Anandhu114

## 📚 Key Concepts Demonstrated

- Unsupervised Machine Learning
- K-Means Clustering
- Exploratory Data Analysis
- Data Preprocessing
- Feature Scaling
- Elbow Method
- Cluster Visualization
- Customer Segmentation

## 📄 License

No license is specified in the provided notebook. If you want others to reuse or modify this project, choose and add an appropriate license file to the repository.

---

If you find this project useful, feel free to star the repository!
