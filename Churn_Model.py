import streamlit as st

# Title
st.title("Customer Churn Prediction Model")

st.divider()
st.markdown("""
This interactive model is based on a class project I did at University of Maryland. 
            
The model was built using **Python** and [Streamlit](https://streamlit.io/).

[View the source code on GitHub](https://github.com/JoeyFromDataPunk/churn_prediction_model/blob/main/Churn_Model.py)
""")

st.divider()

st.markdown("""
### Background:

We will develop a machine learning model that can predict customer churn, or the loss of customers, based on a combination of variables present in our customer data. 

This model will be able to predict which customers may cancel service based on these variables, which will allow us to intervene and potentially retain that customer.

Here is a sample of the raw data:           
            """)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import the dataset
dataset = pd.read_csv("Churn.csv")

# Display data sample
st.dataframe(dataset.head())
st.write("")
st.write("")
st.write("")
st.markdown("""
            After reviewing the data, "area code" is not useful for our purposes, so this will be dropped from future iterations of the data.
            """)
                        
# drop "area code" from the dataset
dataset.drop(['Area code'], axis=1, inplace=True)
dataset['Churn'] = dataset['Churn'].astype(int)

# define our feature columns
feature_columns = ['Account length','Total day minutes', 'Total day calls',
                   'Total day charge', 'Total eve minutes', 'Total eve calls',
                   'Total eve charge', 'Total night minutes', 'Total night calls',
                   'Total night charge','Total intl minutes', 'Total intl calls',
                   'Total intl charge','Customer service calls']

# split the data into X and y arrays
X = dataset[feature_columns]
y = dataset['Churn']

# import our sci-kit modules that will be used to build the model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# encode our labels:
le = LabelEncoder()
y = le.fit_transform(y)

# create the training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# scale our features so that they can all be uniformly evaluated
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

st.divider()

# --- Data Visualizations ---
# these will be used to determine if some features are more relevant than others
# we will start with a Parallel Coordinates Plot
st.write("")
st.write("")
st.markdown("""
            ### Data Visualization 1: Parallel Coordinates Plot
            First, we will create a series of visualizations that will help us narrow down the features we will use to build our machine learning models. First, we will start with a Parallel Coordinates Plot.
            """)

# bringing in a new instance of StandardScaler to avoid contaminating the one we used on X_train
from sklearn.preprocessing import StandardScaler

# create a dataset copy for visualization
scaled_data = dataset.copy()

# fit a new scaler on all feature columns
viz_scaler = StandardScaler()
scaled_features = viz_scaler.fit_transform(scaled_data[feature_columns])
scaled_data[feature_columns] = scaled_features

# create the parallel coordinates plot
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(15,10))
parallel_coordinates(scaled_data[['Churn'] + feature_columns], "Churn", colormap='coolwarm')
plt.title('Parallel Coordinates Plot (Scaled)', fontsize=20)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Non-Churn (0)', 'Churn (1)'],loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.markdown("""
A parallel coordinates plot is a visualization method for displaying data with multiple quantifying variables. Each data point is represented as a line traversing a set of parallel axes; each axis respresents a different variable.

This kind of plot allows us to compare multiple variables simultaneously, which makes it a good starting point for this exercise. 
            
We are looking for features that are more likely to influence churn. "Churned" customers are represented by red lines, so we are looking for peaks with dense red lines at the top. When the peaks are blue, this indicates areas where non-churning customers cluster.

Based on this, it looks like the five strongest features to concentrate on are:

- Total day minutes
- Total day charge
- Total eve minutes
- Total eve charge
- Customer service calls

We also have weaker features that aren't as strong as the first five but may be useful for comparison:

- Total night minutes
- Total night charge
- Total intl minutes
- Total intl charge           
        """)
st.divider()

st.markdown("""
            ### Data Visualization 2: PairPlot
            Our goal is to have more than one model so that we can compare and choose the best fitting model for our predictions. We have isolated five features to use in a model, but we can run additional visualizations to see if there is more opportunity to narrow those features down. Next, we will create a pairplot.
            
            """)
st.write("")
st.markdown("""
            To see this image clearer, right click the plot and select "Open Image In New Tab"
            """)

# We should run a new StandardScaler instance on our five features.
# First, let's list those features in a new variable

top_5_features = ['Total day minutes', 'Total day charge', 'Total eve minutes', 'Total eve charge', 'Customer service calls']

# Second, create a new, smaller dataset that includes Churn
pairplot_data_5 = dataset[top_5_features + ['Churn']].copy()

# Next, we will scale the new dataset
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# create a new scaler
scaler_5 = StandardScaler()
pairplot_data_5[top_5_features] = scaler_5.fit_transform(pairplot_data_5[top_5_features])

# create the pairplot
plt.figure()
sns.pairplot(pairplot_data_5, hue = "Churn", height=3, aspect=1, markers=["o", "s", "D"])
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.write("")
st.markdown("""
            ### Interpretation:

##### 1. Minutes/Charges

The Day and Evening minutes and charges plots form diagonal lines. This tells us that there's a strong correlation between these pairs, which makes sense: if minutes go up, charges should go up too. We can probably pick one (minutes or charges) and eliminate the other to avoid redundancy.

Based on this visualization, it appears that the separation in churn is less defined for these features:

- Total eve minutes
- Total eve charge

However, there is a stronger color separation in these features:

- Total day minutes
- Total day charge
- Customer service calls


##### 2. Customer service calls

There’s a noticeable pattern here: clusters of red at higher values, especially 3 or more service calls. This pattern seems to be present in all pairs of features. This suggests that frustrated customers contacting support multiple times may be more likely to leave.

Something to consider: maybe Customer service calls, on their own, are a stronger predictor of churn than any combination of variables together.
""")

st.divider()

st.write("")           
st.markdown("""
            ### Data Visualization 3: Heatmaps
            
We can use heatmaps to quantify the correlations we see in these plots. This gives us a better picture of:

1. Which set of features (minutes or charges) has a stronger correlation with Churn? This tells us that the other set can be removed for redundancy.

2. How strong of a correlation do Customer service calls have with Churn? This could indicate whether or not a 1-feature model may be just as useful as combinations of features.  

Based on this, we will create three models for heatmapping:

- Model A: All 14 features. Gives us a baseline for the other models.

- Model B: all sets of minutes and charges vs Churn (which set gives the stronger correlation?).
Answers Question 1 above. This will be split into two heatmaps, one for minutes and one for charges.

- Model C: Customer service alone vs Churn
Answers Question 2 above.
""")
st.write("")
st.write("")

# No new scaler needed here
# Heatmaps measure a rate of correlation between -1 and 1 for all values, so scaling is not necessary

st.markdown("""
            ### Model A Heatmap: All 14 Features + Churn)
            """)
# --- Model A heatmap: 14 features + churn ---
model_a_numeric = dataset.select_dtypes(include=[np.number]).columns.tolist()
    
# create dataset from list
model_a = dataset[model_a_numeric]
    
plt.figure(figsize=(18, 12))
sns.heatmap(model_a.corr(), annot=True, cmap="RdYlGn")
plt.title("Heatmap: Model A - 13 features", fontsize=12)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.write("")
st.write("")
st.write("")
st.markdown("""
            ### Model B.1 Heatmap: Minutes + Churn)
            """)
# --- Model B heatmap: minutes + churn ---

# Select only minutes-related features plus Churn
model_b_minutes = dataset.filter([
    'Total day minutes',
    'Total eve minutes',
    'Total night minutes',
    'Total intl minutes',
    'Churn'
], axis=1)

# Plot heatmap for minutes
plt.figure(figsize=(10, 6))
sns.heatmap(model_b_minutes.corr(), annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Heatmap: Model B.1 – Minutes vs Churn", fontsize=12)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.write("")
st.write("")
st.write("")
st.markdown("""
            ### Model B.2 Heatmap: Charges + Churn)
            """)
# --- Model B.2 heatmap: charges + churn ---

# Select only charges-related features plus Churn
model_b_charges = dataset.filter([
    'Total day charge',
    'Total eve charge',
    'Total night charge',
    'Total intl charge',
    'Churn'
], axis=1)

# Plot heatmap for charges
plt.figure(figsize=(10, 6))
sns.heatmap(model_b_charges.corr(), annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Heatmap: Model B.2 – Charges vs Churn", fontsize=12)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.write("")
st.write("")
st.write("")
st.markdown("""
            ### Model C Heatmap: Customer Service Calls + Churn)
            """)
# --- Model C: Customer service calls + Churn ---

# Select only Customer service calls and Churn
model_c = dataset.filter([
    'Customer service calls',
    'Churn'
], axis=1)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(model_c.corr(), annot=True, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Heatmap: Model C – Customer Service Calls vs Churn", fontsize=12)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

st.write("")
st.write("")
st.write("")
st.markdown("""
            ## Interpretation

#### Question 1. Which set of features (minutes or charges) has a stronger correlation with Churn?

Both Model B heatmaps have identical values, so we can safely remove either one. 

##### Conclusion: From a customer service perspective, it makes more sense to eliminate minutes and focus on charges.


#### Question 2. How strong of a correlation does Customer service calls have with Churn?

The Model C heatmap shows us that Customer Service Calls have the second highest correlation to Churn (0.23), below Day Charges (0.24).

##### Conclusion: Customer Service Calls is a valuable feature and could be used alone or combined with other features.

From here, we will build off of these findings by creating four new models and testing them, using K-Nearest Neighbors and cross-validation:

- Model A2 = Customer service + churn
- Model B2 = Customer service + Day charges + churn
- Model C2 = Customer service + Day charges + Evening charges + churn
- Model D2 = Customer service + Day, Evening, Night, International charges + churn 
            """)

st.divider()
st.subheader("K-Nearest Neighbors (KNN)")
st.markdown("""
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for both classification and regression tasks. The basis of KNN is that within a set space, points that are closer to each other are more similar than points that are further away. Based on this assumption, KNN looks for the nearest known datapoints, or “neighbors” and uses their values to predict the value of a new datapoint. For this exercise, we will create multiple models with the data and choose the one most suited for predicting the loss of a customer.
            
We will evaluate each model using confusion matrices and classification reports, as seen above. A confusion matrix tells us how many errors were in the model and where those errors occurred. A classification report gives us four metrics to determine model performance:

    •	Recall – How well the model finds all actual positives (churners).
    •	Precision – How many of the predicted positives were actually correct.
    •	F1-Score – A weighted average of recall and precision.
    •	Accuracy – The overall proportion of correct predictions.

In the next section, we use 10-fold Cross-Validation to measure how each model performs across different data splits, and tune the k-value (number of neighbors) to minimize misclassification error. The graph below shows error rates by k, helping us select the optimal number of neighbors.

            """)
st.divider()

st.markdown("""
            ##### First Model: A2 = Customer Service Calls + Churn
            """)

with st.echo(code_location="below"):
    # --- Model A2 = Customer service + churn ---             
    # Create the model
    model_a2 = dataset[['Customer service calls', 'Churn']].copy()

with st.echo(code_location="below"):
    # Import libraries
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_score
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

with st.echo(code_location="below"):
    # Step 1: Split data
    # We split the data into inputs (X) and target outcome (y)
    # Then divide the data into an 80/20 split of training and testing sets
    X = model_a2.drop('Churn', axis=1)
    y = model_a2['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with st.echo(code_location="below"):
    # Step 2: Fit model with k=3
    # Create and train a KNN classifier using a starting point of k=3 neighbors
    # Use the training data to fit the model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

# Step 3: Evaluation - create confusion matrix 
# Confusion Matrix for A2
# Create two columns for Confusion Matrix
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

# Column 1: Subtitle
with col1:
    st.markdown("""
                Confusion Matrix: Model A2
                """)
#Column 2: Matrix
with col2:
    cm_A2 = confusion_matrix(y_test, y_pred)
    cm_df_A2 = pd.DataFrame(cm_A2, index=['Actual: Not Churn', 'Actual: Churn'], columns=['Predicted: Not Churn', 'Predicted: Churn'])
    st.dataframe(cm_df_A2)

# Create two columns for Crosstab of Predictions
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
# Column 1: Subtitle
with col1:
    st.markdown("""
                Crosstab of Predictions: Model A2
                """)

with col2:
    # Create the crosstab
    ct = pd.crosstab(y_test, y_pred, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True)

    # Display in Streamlit
    st.dataframe(ct)

# Create two columns for Classification Report
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
# Column 1: Subtitle
with col1:
    st.markdown("""
                Classification Report: Model A2
                """)

with col2:
    # Create the report
    report = classification_report(y_test, y_pred, output_dict=True)
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    # Display in Streamlit
    st.dataframe(report_df.style.format("{:.2f}"))
    st.write("")

st.subheader('The overall accuracy of Model A2 is: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')
st.write("")
st.markdown("""
        Next, we will cross-validate and tune the model, then create a misclassification error plot.   
            """)

with st.echo(code_location="below"):
# Step 4: Cross-validation & tuning
# Loop through a list of odd K-values for KNN (odd values prevent ties)
    k_list = list(range(1, 50, 2))
# Create a list of crossvalidation scores; this calculates the average accuracy for each K
    cv_scores = []

with st.echo(code_location="below"):
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

with st.echo(code_location="below"):
# Step 5: Misclassification error plot ---
# This converts cross-validation accuracy to misclassification error
# This gives us the number of incorrect predictions made
    mse = [1 - x for x in cv_scores]

st.write("")
# Plot misclassification error vs. k to visualize which k performs best
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.plot(k_list, mse, marker='o', linestyle='-', color='red')
plt.title("Misclassification Error vs. k (Model A2)", fontsize=16)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Misclassification Error")
plt.xticks(k_list)
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

# --- Step 6: Best k ---
# Find and print the k value with the lowest misclassification error
# This k will be used for future model tuning
best_k = k_list[mse.index(min(mse))]
st.subheader("The optimal number of neighbors for Model A2 is %d." % best_k)

st.write("")
st.markdown("""
        We will now repeat this with the other models.   
            """)
st.write("")

st.markdown("""
            #### Second Model: B2 = Customer Service Calls + Day charges + Churn
            """)
# --- Model B2 = Customer service + Day charges + churn ---

# Create the model
model_b2 = dataset[['Customer service calls', 'Total day charge', 'Churn']].copy()

# Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Split data
X = model_b2.drop('Churn', axis=1)
y = model_b2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 3: Evaluation
# Create two columns for Confusion Matrix
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

# Column 1: Subtitle
with col1:
    st.markdown("""
                Confusion Matrix: Model B2
                """)
#Column 2: Matrix
with col2:
    cm_B2 = confusion_matrix(y_test, y_pred)
    cm_df_B2 = pd.DataFrame(cm_B2, index=['Actual: Not Churn', 'Actual: Churn'], columns=['Predicted: Not Churn', 'Predicted: Churn'])
    st.dataframe(cm_df_B2)

# st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# st.write("\nCrosstab:\n", pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
# st.write("\nClassification Report:\n", classification_report(y_test, y_pred))
# st.write('Accuracy: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# Create two columns for Crosstab of Predictions
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
# Column 1: Subtitle
with col1:
    st.markdown("""
                Crosstab of Predictions: Model B2
                """)

with col2:
    # Create the crosstab
    ct = pd.crosstab(y_test, y_pred, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True)

    # Display in Streamlit
    st.dataframe(ct)

st.subheader('The overall accuracy of Model B2 is: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# Step 4: Cross-validation & tuning
k_list = list(range(1, 50, 2))
# Create a list of crossvalidation scores; this calculates the average accuracy for each K
cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Step 5: Misclassification error plot ---
# This converts cross-validation accuracy to misclassification error
# This gives us the number of incorrect predictions made
mse = [1 - x for x in cv_scores]

# Plot misclassification error vs. k to visualize which k performs best
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.plot(k_list, mse, marker='o', linestyle='-', color='red')
plt.title("Misclassification Error vs. k (Model B2)", fontsize=16)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Misclassification Error")
plt.xticks(k_list)
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

# --- Step 6: Best k ---
# Find and print the k value with the lowest misclassification error
# This k will be used for future model tuning
best_k = k_list[mse.index(min(mse))]
st.subheader("The optimal number of neighbors for Model B2 is %d." % best_k)

st.write("")

st.markdown("""
            #### Third Model: C2 = Customer Service Calls + Day charges + Evening charges + Churn
            """)

# --- Model C2 = Customer service + Day charges + Evening charges + churn ---

# Create the model
model_c2 = dataset[['Customer service calls', 'Total day charge', 'Total eve charge', 'Churn']].copy()

# Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Split data
X = model_c2.drop('Churn', axis=1)
y = model_c2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 3: Evaluation
# Create two columns for Confusion Matrix
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

# Column 1: Subtitle
with col1:
    st.markdown("""
                Confusion Matrix: Model C2
                """)
#Column 2: Matrix
with col2:
    cm_C2 = confusion_matrix(y_test, y_pred)
    cm_df_C2 = pd.DataFrame(cm_C2, index=['Actual: Not Churn', 'Actual: Churn'], columns=['Predicted: Not Churn', 'Predicted: Churn'])
    st.dataframe(cm_df_C2)

# Create two columns for Crosstab of Predictions
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
# Column 1: Subtitle
with col1:
    st.markdown("""
                Crosstab of Predictions: Model C2
                """)

with col2:
    # Create the crosstab
    ct = pd.crosstab(y_test, y_pred, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True)

    # Display in Streamlit
    st.dataframe(ct)

st.subheader('The overall accuracy of Model C2 is: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# st.write("\nCrosstab:\n", pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
# st.write("\nClassification Report:\n", classification_report(y_test, y_pred))
# st.write('Accuracy: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# Step 4: Cross-validation & tuning
k_list = list(range(1, 50, 2))
# Create a list of crossvalidation scores; this calculates the average accuracy for each K
cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Step 5: Misclassification error plot ---
# This converts cross-validation accuracy to misclassification error
# This gives us the number of incorrect predictions made
mse = [1 - x for x in cv_scores]

# Plot misclassification error vs. k to visualize which k performs best
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.plot(k_list, mse, marker='o', linestyle='-', color='red')
plt.title("Misclassification Error vs. k (Model C2)", fontsize=16)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Misclassification Error")
plt.xticks(k_list)
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

# --- Step 6: Best k ---
# Find and print the k value with the lowest misclassification error
# This k will be used for future model tuning
best_k = k_list[mse.index(min(mse))]
st.subheader("The optimal number of neighbors for Model C2 is %d." % best_k)

st.write("")

st.markdown("""
            #### Fourth Model: D2 = Customer Service Calls + Day, Evening, Night, International charges + Churn
            """)

# --- Model D2 = Customer service + Day, Evening, Night, International charges + churn ---

# Create the model
model_d2 = dataset[['Customer service calls', 'Total day charge', 'Total eve charge', 
                    'Total night charge','Total intl charge','Churn']].copy()

# Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Split data
X = model_d2.drop('Churn', axis=1)
y = model_d2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 3: Evaluation
# Create two columns for Confusion Matrix
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

# Column 1: Subtitle
with col1:
    st.markdown("""
                Confusion Matrix: Model D2
                """)
#Column 2: Matrix
with col2:
    cm_D2 = confusion_matrix(y_test, y_pred)
    cm_df_D2 = pd.DataFrame(cm_D2, index=['Actual: Not Churn', 'Actual: Churn'], columns=['Predicted: Not Churn', 'Predicted: Churn'])
    st.dataframe(cm_df_D2)

# Create two columns for Crosstab of Predictions
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed
# Column 1: Subtitle
with col1:
    st.markdown("""
                Crosstab of Predictions: Model D2
                """)

with col2:
    # Create the crosstab
    ct = pd.crosstab(y_test, y_pred, 
                 rownames=['Actual'], 
                 colnames=['Predicted'], 
                 margins=True)

    # Display in Streamlit
    st.dataframe(ct)

st.subheader('The overall accuracy of Model D2 is: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# st.write("\nCrosstab:\n", pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
# st.write("\nClassification Report:\n", classification_report(y_test, y_pred))
# st.write('Accuracy: ' + str(round(accuracy_score(y_test, y_pred),2) * 100) + '%')

# Step 4: Cross-validation & tuning
k_list = list(range(1, 50, 2))
# Create a list of crossvalidation scores; this calculates the average accuracy for each K
cv_scores = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Step 5: Misclassification error plot ---
# This converts cross-validation accuracy to misclassification error
# This gives us the number of incorrect predictions made
mse = [1 - x for x in cv_scores]

# Plot misclassification error vs. k to visualize which k performs best
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plt.plot(k_list, mse, marker='o', linestyle='-', color='red')
plt.title("Misclassification Error vs. k (Model D2)", fontsize=16)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Misclassification Error")
plt.xticks(k_list)
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()  # Clear the figure so it doesn't stack

# --- Step 6: Best k ---
# Find and print the k value with the lowest misclassification error
# This k will be used for future model tuning
best_k = k_list[mse.index(min(mse))]
st.subheader("The optimal number of neighbors for Model D2 is %d." % best_k)

st.markdown("""
            ## Analysis:

For this exercise, I will be focusing on two metrics: Recall and Overall Accuracy. Here is why:

- Recall tells us how many times the model correctly identified churners vs non-churners. 

In this case, this is extremely important because if the model has a low Recall for churners, it means that we will miss opportunities to intervene and retain those customers. Missing churners are lost customers that we may have been able to save. 

Additionally, if the model has a low Recall for non-churners, it means that we may be intervening with customers who had no plans to leave. For this reason, we are looking for the highest Recall of 1 (Churn). 

- Accuracy gives us a high-level summary of the overall performance of the model.

We should be looking at Recall first for this exercise, specifically because we need to identify churners as accurately as possible. When Recall is comparable across models, we can use Accuracy as the tie-breaker. This is because a high Accuracy score alone could be misleading, as it may just mean that the model is good at predicting non-churners, which is not our goal. 

For this same reason, we would not want to rely on Precision as a metric. Precision tells us how good the model is at identifying results within each class. In other words: 

- Recall answers the question: Of all the actual churners, how many did the model correctly identify?
- Precision answers the question: Of all the churners we identified, how many actually churned?

If this was an exercise where we needed to prioritize false positives AND false negatives, we may also want to consider Precision and the F1-score (a weighted average of precision and recall).

However, we are only considering Recall of Class 1 and Accuracy as our main metrics for success. The results of the models were:

- A2 had high accuracy (91%) but low Class 1 Recall (27%), making it the least effective model of the four.
- B2, C2, and D2 had identical Class 1 Recall. However, B2 had considerably higher accuracy (93%).

### Based on these results, Model B2 (Customer Service Calls, Day Charges, Churn) would be the most effective predictor of churn.
            
### Try it yourself! Using the interface below, enter the number of customer service calls and day charges, and Model B2 will predict whether or not the customer will stay or leave.
            """)

# Re-train Model B2 on the full dataset for final prediction
final_X = model_b2.drop('Churn', axis=1)
final_y = model_b2['Churn']

final_scaler = StandardScaler()
final_X_scaled = final_scaler.fit_transform(final_X)

final_knn = KNeighborsClassifier(n_neighbors=5)
final_knn.fit(final_X_scaled, final_y)

# Add a prediction interface
st.markdown("## Predict Customer Churn")

# User inputs
cust_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, max_value=20, value=1)
day_charge = st.number_input("Total Day Charge ($)", min_value=0.0, max_value=100.0, value=30.0)

# Create input array
user_input = np.array([[cust_service_calls, day_charge]])

# Scale the input
user_input_scaled = final_scaler.transform(user_input)

# Make prediction
prediction = final_knn.predict(user_input_scaled)

# Output the result
if prediction[0] == 1:
    st.error("⚠️ This customer is predicted to leave (churn).")
else:
    st.success("✅ This customer is predicted to stay.")

# streamlit run Churn_Model.py