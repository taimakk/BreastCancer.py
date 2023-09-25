import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
df['Diagnosis'] = breast_cancer.target

# Streamlit App
st.title("Breast Cancer Diagnosis Visualization")

# Display a sample of the dataset
st.subheader("Sample of the Dataset")
st.write(df.head())

# Create a bar chart to visualize the distribution of diagnosis (Malignant or Benign)
st.subheader("Diagnosis Distribution")
diagnosis_counts = df['Diagnosis'].value_counts().rename({0: "Malignant", 1: "Benign"})
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, ax=ax)
plt.xlabel("Diagnosis")
plt.ylabel("Count")
st.pyplot(fig)

# Explanation
st.markdown("This is a simple Streamlit app that visualizes the distribution of breast cancer diagnosis.")

# Pair Plot
st.subheader("Pair Plot of Selected Features")
selected_features = st.multiselect("Select features for pair plot:", df.columns[:-1])
show_grid = st.checkbox("Show Grid Lines", value=True)  # Checkbox to show/hide grid lines

if selected_features:
    # Create pair plot with or without grid lines based on the checkbox value
    pair_plot = sns.pairplot(df, hue='Diagnosis', vars=selected_features, diag_kind="kde", plot_kws={"grid": show_grid})
    st.pyplot(pair_plot.fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
heatmap_fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(heatmap_fig)

# Box Plot for Feature Distribution
st.subheader("Box Plot of Selected Features")
selected_feature = st.selectbox("Select a feature for the box plot:", df.columns[:-1])
num_bins = st.slider("Number of Bins", min_value=5, max_value=50, value=20)  # Slider for number of bins

plt.figure(figsize=(8, 6))
box_plot_fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(x=df[selected_feature], hue=df['Diagnosis'], bins=num_bins, kde=True, ax=ax)
st.pyplot(box_plot_fig)

# Count Plot for Diagnosis
st.subheader("Count Plot for Diagnosis")
plt.figure(figsize=(6, 4))
count_plot_fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=df, ax=ax)
st.pyplot(count_plot_fig)




