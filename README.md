# 1 Breast cancer prevention

Predicting if a breast cancer is benign or malignant

The goal is to train a classification model based on a dataset of cancer cell features to predict whether a given cell belongs to a benign or malignant tumor.

The dataset was created by the University of Wisconsin. It considered 569 cases (rows - samples) and 32 attributes (columns - features).

# Dataset information and understanding 

## Why was this type of analysis chosen? 

![App Screenshot](breast_cancer_stages.png)

## Dataset

The parameters under examination were obtained from digital images of breast nodule samples taken with needle biopsy. They describe the characteristics of the cell nuclei present in each image.

(Needle biopsy = diagnostic procedure in which a sample of a mass suspected of being pathological is taken with a thin needle, in order to analyze it under a microscope in the laboratory)

![App Screenshot](breast_in medical_visit.png)

Dataset attribute information:

-  ID number

- Diagnosis (M = Malignant; B = Benign)

3-32) Ten features obtained for each cell nucleus:
- radius (average of the distances of the perimeter points from the center);
- texture;
- perimeter;
- area;
- smoothness (local variation of the radius length);
- compactness (perimeter^2 / area - 1);
- concavity (amount of concave portions of the edge);
- concavity points (number of concave portions of the edge);
- symmetry;
- fractal dimension ("coastline approximation" - 1, is the degree of irregularity and fragmentation).

For each image, "mean" (average), "se" (standard error) and "worst" (average of the 3 largest values) of each parameter were calculated. The result is that there are 30 features.
For example, in column #2 we find "radius_mean", in column #12 "radius_se" and in column #22 "radius_worst".

Each attribute is indicated with four significant digits.
## 2 Import the necessary libraries

you can load them with the following command or by running the program directly (since I have set the automatic installation of the missing libraries)

```python library
pip install -r requirements.txt
```

# 3 Load the dataset

The dataset must be downloaded locally and uploaded using the command below. The file to download is available at: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

## 4 Let's look at the data

```python commands
data.head()

data.tail()

data.info()
```

We notice that there is a column "Unnamed: 32" that contains null values: we are going to delete it. We also delete the column that contains the IDs of the samples, it is not useful for classifying the data.

"diagnosis" is our class label, that is, the attribute whose value we want to predict based on the other attributes.

We also notice that the remaining data is all in float format, except for the diagnosis, which we are going to convert:

```python commands
data = data.drop(['Unnamed: 32','id'],axis = 1)
data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})

data_save=data
```

we save the cleaned df in a variable to be able to retrieve it later and work more easily

We print the first 5 lines (head) again to observe the data.

```python commands
data.head()
```









# 5 Relationships between variables

This is the simplest way to study the relationship between variables.

For example, let's see if there is a relationship between the mean radius and the mean area.

```python commands
data = data_save
plt.figure(figsize=(15, 10))
sns.jointplot(x='radius_mean', y='area_mean', data=data, kind='reg')
plt.show()
```

Of course we can observe a correlation: when the average radius increases, the average area also increases. So there is a positive correlation between the two.

X

Conversely, we note that there is no correlation between the mean area and the standard error of the fractal dimension: the standard error of the fractal dimension is not affected by the variation of the area.

```python commands
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
```

There are almost perfect linear patterns between radius, perimeter, and area. This indicates that there is multicollinearity between these variables (i.e. they are highly correlated linearly). Another set of variables that could imply multicollinearity are concavity, concave points, and compactness.

Multicollinearity is a problem, as it could undermine the significance of the independent variables. We address this by removing highly correlated predictors from the model.

We can examine the strength of the relationships between pairs of variables. We say that two variables are related to each other, if one gives us information about the other.

Let's observe the correlation between all the available characteristics in the matrix. The range of obtainable numbers goes from 1 (maximum positive correlation between two variables) to -1 (maximum negative correlation between two variables). A value of 0 indicates that there is no correlation between the two variables.

```python commands
f,ax=plt.subplots(figsize = (20,20)) 
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".2f",ax=ax) 
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()
```

We can check for multicollinearity between some of the variables. For example, the mean radius has a correlation of 1 and 0.99 with the mean perimeter and mean area, respectively. This is because the three columns essentially contain the same information, which is the physical size of what we are looking at (the cell nucleus).

Therefore, we will only select one of these three features as we continue with the analysis.

We also see multicollinearity between the "mean" and "worst" columns. For example, the mean radius has a correlation of 0.97 with the worst radius.

Additionally, there is multicollinearity between the attributes: compactness, concavity, and concavity points. So we can only select one of these. Let's choose compactness.

```python commands
cols = ['radius_worst',
        'texture_worst',
        'perimeter_worst',
        'area_worst',
        'smoothness_worst',
        'compactness_worst',
        'concavity_worst',
        'concave points_worst',
        'symmetry_worst',
        'fractal_dimension_worst']
data = data.drop(cols, axis=1)


cols = ['perimeter_mean',
        'perimeter_se',
        'area_mean',
        'area_se']
data = data.drop(cols, axis=1)


cols = ['concavity_mean',
        'concavity_se',
        'concave points_mean',
        'concave points_se']
data = data.drop(cols, axis=1)

data_final = data

data.columns

f,ax=plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".2f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()
```
# 6. Data visualization

## Histogram

Let's look at the number of benign B and malignant M tumors in the dataset.

```python commands
data = data_save
frequencies = data['diagnosis'].value_counts()

plt.bar(frequencies.index, frequencies.values)

colors = ['green', 'red'] 
plt.bar(frequencies.index, frequencies.values, color=colors)

plt.xticks(frequencies.index, ['Benigni', 'Maligni'])


plt.xlabel('Valori')
plt.ylabel('Frequenza')
plt.title('Istogramma della variabile dicotomica')

plt.show()

lista = ["radius_mean", "texture_mean", 'smoothness_mean',
       'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
       'symmetry_se', 'fractal_dimension_se']


for i in lista:
  m = plt.hist(data[data["diagnosis"] == 1][i], bins=30, fc=(1,0,0,0.5), label="Tumori Maligni")
  b = plt.hist(data[data["diagnosis"] == 0][i], bins=30, fc=(0,1,0,0.5), label="Tumori Benigni")
  plt.legend()
  plt.xlabel(f"{i}")
  plt.ylabel("Frequenza")
  plt.title("Istogramma del {} del nucleo di cellule tumorali".format(i))
  plt.show()
  frequent_malignant_radius_mean = m[0].max()
```

Let's start by studying the distribution of the variables in the simplest way: the histogram allows us to visualize how many times each value appears in the dataset (frequency).

We can go and observe some parameters in particular: despite there being a slight overlap, we can see that most of the cells of benign tumors have a radius (and therefore an area) smaller than those of malignant tumors. The same goes for concavity.

The radius and area distributions concerning benign tumors can be approximated to a normal one.

From the graphs, we can see that the average radius and compactness of the nuclei of cells belonging to malignant tumors tend to be greater than those in benign tumors.

## curve 

Let's draw the curves for the parameters we want to study in depth:

```python commands
for i in lista:
  data[i].plot.kde(title = i)
  plt.show()
```

We can observe in the mean radius curve a positive kurtosis and a positive skewness.

(kurtosis = departure from distribution normality, with respect to which there is a greater flattening or a greater lengthening. Its most well-known measure is the Pearson index)

Clearly this is due to the fact that we have less data regarding malignant tumors (see histogram 1). The second peak in the radius graph is due to the fact that I am simultaneously observing the data of benign and malignant tumors, not to the presence of outliers. We can see it better by separating the two curves:

```python commands
for i in lista:
  data[i][data["diagnosis"] == 1].plot.kde(title = i, c = "r")
  data[i][data["diagnosis"] == 0].plot.kde(title = i, c = "g")
  plt.show()
```

## PCA 

These below are the cells that are at the end of the chapter of colinearity, the meaning of PCA must be better explained.

For PCA, we will need a y containing the diagnosis values, while such values ​​must not be present in the original dataset.

```python commands
y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

x.head()


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


pca = PCA(n_components=5) 
pca.fit(x_scaled) 
variance_ratio = pca.explained_variance_ratio_

plt.bar(range(len(variance_ratio)), variance_ratio)
plt.xlabel('Componenti Principali')
plt.ylabel('Percentuale di varianza spiegata')

for i, val in enumerate(variance_ratio):
    plt.text(i, val, str(round(val*100, 2)) + '%') 

plt.show()
```

The first 3 principal components alone explain more than 85% of the significance of the data. We can then take these first three principal components and graph them, obtaining a 3D scatter plot.

```python commands
pca = PCA(n_components = 3)
pca.fit(x_scaled) 
X_reduced_pca = pca.transform(x_scaled) 

pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2","p3"]) 
pca_data["diagnosis"] = y

hue = pca_data["diagnosis"]
data = [
    go.Scatter3d(
        x=pca_data.p1,
        y=pca_data.p2,
        z=pca_data.p3,
        mode='markers',
        marker=dict(
            size=4,
            color=hue,
            symbol="circle",
            line=dict(width=2)
        )
    )
]

layout = go.Layout(title="PCA",
                   scene=dict(
                       xaxis=dict(title="p1"),
                       yaxis=dict(title="p2"),
                       zaxis=dict(title="p3")
                   ),
                   hovermode="closest")

fig = go.Figure(data=data, layout=layout)


fig.update_layout(scene=dict(camera=dict(up=dict(x=0, y=0, z=1),
                                         center=dict(x=0, y=0, z=0),
                                         eye=dict(x=2, y=2, z=0.1))),
                  updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    buttons=[dict(label='Rotate',
                                                  method='animate',
                                                  args=[None,
                                                        dict(frame=dict(duration=50, redraw=True),
                                                             fromcurrent=True,
                                                             transition=dict(duration=0))
                                                       ]
                                                 )
                                            ]
                                   )
                             ])

pyo.iplot(fig)
```

we can clearly see in the graph how two clusters are distinguished, with even more significance given by the third principal component. We can however appreciate a significant division into clusters from a 2D scatter plot.

```python commands
data=data_save
y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

# la PCA ha bisogno della standardizzazione dei dati
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Costruzione PCA
pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["diagnosis"] = y
hue =pca_data["diagnosis"]
data = [go.Scatter(x = pca_data.p1,
                   y = pca_data.p2,
                   mode = 'markers',
                   marker=dict(
                           size=12,
                           color=hue,
                           symbol="pentagon",
                           line=dict(width=2)
                           ))]

layout = go.Layout(title="PCA",
                   xaxis=dict(title="p1"),
                   yaxis=dict(title="p2"),
                   hovermode="closest")
fig = go.Figure(data=data,layout=layout)
pyo.iplot(fig)
```

## Outliers

Looking at the distribution histogram of the mean areas, we notice the presence of some rarer values ​​in the two distributions. These values ​​can be due to errors or rare events and are called "outliers".

To calculate the outliers you need to:

 * calculate the first quartile (Q1)(25%);
 * find the IQR (Inter Quartile Range) = Q3 - Q1;
 * calculate Q1 - 1.5 IQR and Q3 + 1.5 IQR;
 * every value outside this range is an outlier.

```python commands
data = data_save

data_bening = data[data["diagnosis"] == 0]
data_bening.drop('diagnosis', inplace=True, axis=1)
list_column = data_bening.columns

for element in list_column:
  print("\n" + element.upper())
  data_bening = data[data["diagnosis"] == 0]
  desc = data_bening[element].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("\nPer i tumori benigni, ogni valore fuori da questo range è un outlier: (", lower_bound ,",", upper_bound,")")
  data_bening[data_bening[element] < lower_bound][element]
  print("Outliers: ", data_bening[(data_bening[element] < lower_bound) | (data_bening[element] > upper_bound)][element].values)

  data_malignant = data[data["diagnosis"] == 1]
  data_malignant.drop('diagnosis', inplace=True, axis=1)
  data_malignant.head()
  list_column = data_malignant.columns

  for element in list_column:
  print("\n" + element.upper())
  data_malignat = data[data["diagnosis"] == 1]
  desc = data_malignant[element].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("\nPer i tumori maligni, ogni valore fuori da questo range è un outlier: (", lower_bound ,",", upper_bound,")")
  data_malignant[data_malignant[element] < lower_bound][element]
  print("Outliers: ", data_malignant[(data_malignant[element] < lower_bound) | (data_malignant[element] > upper_bound)][element].values)  

  from sklearn.cluster import DBSCAN

# Apply DBSCAN to the PCA data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_reduced_pca)

# Get the outliers identified by DBSCAN
outliers = pca_data[dbscan.labels_ == -1]

# Visualize the PCA with outliers highlighted
data = [go.Scatter(x = pca_data.p1,
                   y = pca_data.p2,
                   mode = 'markers',
                   marker=dict(
                           size=12,
                           color=hue,
                           symbol="pentagon",
                           line=dict(width=2)
                           )),
        go.Scatter(x = outliers.p1,
                   y = outliers.p2,
                   mode = 'markers',
                   marker=dict(
                           size=3,
                           color='red',
                           symbol="circle",
                           line=dict(width=2)
                           ))]

layout = go.Layout(title="PCA with Outliers Detected by DBSCAN",
                   xaxis=dict(title="p1"),
                   yaxis=dict(title="p2"),
                   hovermode="closest")

fig = go.Figure(data=data,layout=layout)
pyo.iplot(fig)
```

## Box Plot 

This is another way of visualizing outliers.

```python commands
data = data_final

data_bening = data[data["diagnosis"] == 0]
data_bening.drop('diagnosis', inplace=True, axis=1)
list_column = data_bening.columns

for element in list_column:
  melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = element)
  plt.figure(figsize = (7,7))
  sns.boxplot(x = "variable" , y = "value", hue="diagnosis",data = melted_data)
  plt.show()
```

## Violin plot 

Violin plots are inspired by box plots with whiskers (boxplots) by reporting for a univariate distribution, instead of the classic boxes, the density profile of the observed values ​​in the form of a kernel density plot (density estimate).

The typical shape of the graphs, which gives them their name, comes from the fact that the kernel density plot of the data is reported symmetrically on both sides of the distribution.

```python commands
y = data["diagnosis"]
x = data.drop(["diagnosis"],axis=1)

data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y, data_n_2], axis=1)

data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
plt.xticks(rotation=90)
```

## Cohen's effect size

Cohen's effect size is a statistical measure that indicates the difference between the means of two groups divided by the common standard deviation of the two groups. Its practical interpretation is as follows: a Cohen's effect size of 0.2 indicates a small difference between the two groups, 0.5 indicates a medium difference, and 0.8 indicates a large difference.

It is one of the statistics used to summarize a set of observations and therefore communicate the greatest amount of information in the simplest way possible.

It describes the size of an effect, that is, the strength of the relationship between two variables.

Let's compare the effect size between the mean of the radius in benign and malignant tumors.

```python commands
data = data_final

gruppo_0 = data.loc[data['diagnosis'] == 0]
gruppo_1 = data.loc[data['diagnosis'] == 1]


colonne = [col for col in data.columns if col != 'diagnosis']


for col in colonne:
    diff_media = gruppo_0[col].mean() - gruppo_1[col].mean()
    std_com = (gruppo_0[col].std() + gruppo_1[col].std()) / 2
    d = abs(diff_media / std_com)
    print("Dimensione dell'effetto di Cohen per la feature", col, ":", d)
```




# 7 Machine Learning Algorithms

We now apply various machine learning models for binary classification of samples into Benign and Malignant tumors.

## Preparazione dati

The dataset contains the 569 instances of breast lesions described by 30 quantitative features. Of these, we selected 12 for analysis. The feature diagnosis represents the class label (M for malignant and B for benign).

The code loads the dataset using pandas, converts the class labels to numeric values, splits the data into training and test sets, normalizes the data, and trains several machine learning models to predict the class (B or M).

Finally, for each model, a model is used to predict the class labels for the test data and the accuracy is calculated.

```python commands
  data = data_final

  X = data.drop(['diagnosis'],axis=1)
  y = data['diagnosis']

  X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

  print(f"Train set:\n{y_train.value_counts()}")

  print(f"Test set:\n{y_test.value_counts()}")

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
```

## KNN

We apply scikitlearn's grid search to find the best hyperparameters for a KNN model:

```python commands
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_search = GridSearchCV(knn, param_grid, cv=5)

grid_search.fit(X_train_scaled, y_train)

print("Iperparametri ottimali:\n", grid_search.best_params_)
```

In this example, a KNN model is created using the KNeighborsClassifier class from the scikit-learn library. A grid of parameter values ​​to test is then defined using the param_grid dictionary. The cv parameter in the GridSearchCV specifies the number of folds for cross-validation. Finally, a search for the best hyperparameters is performed using the fit function of the GridSearchCV object, and the results are printed to the screen using best_params_.

```python commands
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')

model4 = knn.fit(X_train_scaled, y_train)

prediction4 = model4.predict(X_test_scaled)

cm4 = confusion_matrix(y_test, prediction4)
sns.heatmap(cm4,annot=True)
```

In this example, the KNN model is built using the best hyperparameters found (n_neighbors=3, weights='uniform', algorithm='auto'). Next, the model is trained on the training data using the fit method, and finally predictions are made on the test set using the predict method.

```python commands
TP=cm4[0][0]
TN=cm4[1][1]
FN=cm4[1][0]
FP=cm4[0][1]
acc4 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc4)

print(classification_report(y_test, prediction4)) #prediction?
```

We apply scikitlearn's grid search to find the best hyperparameters for an SVM model


```python commands
# Definisci i possibili valori degli iperparametri da testare
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly']}

# Crea un oggetto GridSearchCV per la ricerca degli iperparametri
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Addestra il modello sul train set
grid_search.fit(X_train_scaled, y_train)

# Stampa i migliori iperparametri trovati
print("Iperparametri ottimali:\n", grid_search.best_params_)
```

In this example, the hyperparameter grid to test is defined by the variable param_grid, which includes a list of possible values ​​for the three hyperparameters of the SVM model (C, gamma, and kernel). A GridSearchCV object is then created using an SVM model, the hyperparameter grid, and the 5-fold cross-validation technique (cv=5). The model is trained on the training data using the fit method of the GridSearchCV object, and finally the best hyperparameters found are printed using the best_params_ attribute.

```python commands
# Crea il modello di SVM con i migliori iperparametri
svm = SVC(C=1, gamma=0.1, kernel='rbf')

# Addestrare il modello utilizzando il set di addestramento
model5 = svm.fit(X_train_scaled, y_train)

# Valutare le prestazioni del modello utilizzando il set di test
prediction5 = model5.predict(X_test_scaled)

cm5 = confusion_matrix(y_test, prediction5)
sns.heatmap(cm5,annot=True)

TP=cm5[0][0]
TN=cm5[1][1]
FN=cm5[1][0]
FP=cm5[0][1]
acc5 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc5)

print(classification_report(y_test, prediction5))
```

In this code, an SVC object is created with the best hyperparameters (C=1, gamma=0.1, and kernel='rbf'). The model is then trained on the training data using the fit method, and finally predictions are made on the test set and the accuracy of the model is calculated using the score method.

In this example, the SVM algorithm uses a linear kernel, which is one of the simplest kernels and is suitable for small-to-medium sized datasets such as the Wisconsin Breast Cancer. Furthermore, cross-validation was used to split the dataset into training and testing sets. Finally, the evaluation metrics of accuracy, precision, recall, and F1-score were used to evaluate the performance of the SVM model.

Update This code loads the breast cancer dataset from the sklearn.datasets module, splits the dataset into training and testing sets, normalizes the attributes using the StandardScaler class from the sklearn.preprocessing module, creates an instance of svm.SVC with a linear kernel and regularization constant of 1, trains the model using the training set, makes predictions on the testing set, and calculates the model performance using the accuracy, precision, recall, and F1-score metrics from the sklearn.metrics module.

## Logit Model (or Logistic Regression)

It is a nonlinear regression model used when the dependent variable is dichotomous. The goal of the model is to establish the probability with which an observation can generate one or the other value of the dependent variable; it can also be used to classify observations, based on their characteristics, into two categories.

To perform a grid search to find the best hyperparameters for logistic regression, you must first define a grid of values ​​for each hyperparameter you want to test. Here are the parameters that are usually put into play for a logistic regression:

penalty: specifies the norm used in the regularization. You can test the values ​​"l1" and "l2". C: the inverse regularization parameter, which controls the strength of the regularization. You can test the values ​​in a range of interest, for example [0.1, 1, 10]. solver: specifies the optimization algorithm used to solve the logistic regression problem. You can test the values ​​"lbfgs", "liblinear", "newton-cg", "sag" and "saga". max_iter: The maximum number of iterations allowed for the solver to converge. You can test values ​​in a range of interest, such as [100, 500, 1000]. Once you have defined these hyperparameters and the values ​​to test for each of them, you can perform a grid search to find the optimal parameters for the logistic regression. This can be done for example using the GridSearchCV class from scikit-learn.


```python commands

params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'tol': [1e-3, 1e-4],
          'solver': ['lbfgs', 'liblinear', 'saga']}

logreg=LogisticRegression(max_iter = 1000)

grid_search = GridSearchCV(logreg, params, cv=5)

grid_search.fit(X_train_scaled, y_train)

print("I parametri ottimali sono:", grid_search.best_params_)
print("Lo score di validazione incrociata è:", grid_search.best_score_)

logreg = LogisticRegression(penalty='l2', C=10, solver='lbfgs', tol=0.001)

model1 = logreg.fit(X_train_scaled,y_train)
prediction1 = model1.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,prediction1)
sns.heatmap(cm1,annot=True)

TP=cm1[0][0]
TN=cm1[1][1]
FN=cm1[1][0]
FP=cm1[0][1]
acc1 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc1)

print(classification_report(y_test, prediction1)) # classification report
```

Accuracy is a measure of model performance and indicates the percentage of correct predictions out of the total predictions. In this case, the model accuracy is 94.73%, which means that the model correctly predicted 94.73% of the cases in the test set.

This accuracy value is good, but it may also be useful to evaluate other metrics such as precision, recall, or F1-score, depending on the specific needs of the problem and application domain.

This code prints the classification report containing the precision, recall and F1-score metrics for both classes (0 and 1). The classification report helps you understand how the model is performing for both classes, so you can evaluate whether the model is doing better for one class than the other.

## Decision Tree

To perform a grid search for a decision tree in Scikit-Learn, you specify a grid of values ​​for the decision tree hyperparameters and use the GridSearchCV class to find the best hyperparameters.

In this case, the grid of values ​​includes four decision tree hyperparameters: criterion, max_depth, min_samples_split, and min_samples_leaf. For each hyperparameter, we specified a list of values ​​that we want to test during the grid search.

Once you have defined the grid of values, you can use the GridSearchCV class to perform the grid search, for example:

```python commands
dt = DecisionTreeClassifier(random_state=42,max_depth=3, min_samples_split=10)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_params_)

dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=4)

model2 = dt.fit(X_train_scaled, y_train)

prediction2 = model2.predict(X_test_scaled)

cm2 = confusion_matrix(y_test, prediction2)
sns.heatmap(cm2,annot=True)
```

In this example, we created a new DecisionTreeClassifier object, defined the grid of hyperparameter values, and used the GridSearchCV class to perform the grid search with 5-fold cross-validation. At the end of the grid search, we printed the best hyperparameters found.

Once you find the best hyperparameters, you can use them to train the decision tree model using the same methods you used for logistic regression, for example:

```python commands
TP=cm2[0][0]
TN=cm2[1][1]
FN=cm2[1][0]
FP=cm2[0][1]
acc2 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc2)
```

In this code, we use the DecisionTreeClassifier class from the scikit-learn library to train a decision tree on the training set. We then make predictions on the test set and calculate the accuracy of the predictions using the accuracy_score function from the scikit-learn library.

Remember that the decision tree is a model that can be sensitive to overfitting, so it may be useful to set some parameters to avoid overfitting, for example by setting the max_depth or min_samples_split parameter.

In this example, we set the max_depth parameter to 3 to prevent the tree from growing too deep and becoming too complex. Additionally, we set the min_samples_split parameter to 10 to ensure that an internal node can only be split if it contains at least 10 samples. These parameters can be adjusted to find the right balance between performance and model complexity.

Remember that the choice of parameters depends on the specific dataset, and you may need to experiment with different combinations of parameters to find the optimal ones for your dataset.

```python commands
print(classification_report(y_test, prediction2))
```

In this example, we used the classification_report function to evaluate the performance of the decision tree on the test set. The function takes the true values ​​y_test and the predicted values ​​y_pred as input, and returns a table that summarizes the evaluation metrics for each class (in this case, "malignant" and "benign"). The metrics include precision, recall, F1-score, and support.

Precision indicates the percentage of instances classified as positive that were actually positive. Recall indicates the percentage of positive instances that were correctly classified. The F1-score is a harmonic mean of precision and recall. Support indicates the number of instances of each class in the test set.

Remember that the evaluation metrics depend on the type of problem you are tackling and your specific needs. For example, if your goal is to identify as many positive cases as possible (e.g., malignant tumors), you might focus on recall (which indicates the percentage of positive cases that you identified correctly). If you instead want to minimize false positives (i.e., cases that you classified as positive but are actually negative), you might focus on accuracy.

## Random Forest 

To perform a grid search on the parameters of a Random Forest in Scikit-Learn, you can use the RandomForestClassifier class together with the GridSearchCV class. Here is an example of how to do it:

```python commands
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

print("Iperparametri ottimali:\n", grid_search.best_params_)
```

In this example, n_estimators is the number of trees in the forest, max_depth is the maximum depth of trees, min_samples_split is the minimum number of samples required to split an internal node, min_samples_leaf is the minimum number of samples required in a leaf node, and max_features is the maximum number of features considered for splitting a node.

You can modify the list of hyperparameters param_grid to suit your needs. Once the grid search is done, you can use the optimal parameters to train a Random Forest model and evaluate its performance on the test data.

```python commands
rf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='auto', random_state=42)

model3 = rf.fit(X_train_scaled, y_train)

prediction3 = model3.predict(X_test_scaled)

cm3 = confusion_matrix(y_test, prediction3)
sns.heatmap(cm3,annot=True)

TP=cm3[0][0]
TN=cm3[1][1]
FN=cm3[1][0]
FP=cm3[0][1]
acc3 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc3)

print(classification_report(y_test, prediction3))
```

In this example, X_train_scaled and y_train are the previously split training data, while X_test_scaled is the scaled test data. The model is trained using the best hyperparameters found via grid search, and predictions are made on the test set. Finally, the confusion matrix is ​​visualized as a heatmap using the Seaborn library.

## Gradient Boosting

Gradient Boosting is a machine learning algorithm that is based on training a sequence of decision tree models, each of which tries to improve the performance of the previous model.

Here is an example of how to perform Grid Search to find the best parameters for the Gradient Boosting model:

```python commands
gb = GradientBoostingClassifier()

params = {'learning_rate': [0.05, 0.1, 0.2],
          'n_estimators': [50, 100, 200],
          'max_depth': [2, 3, 4]}

grid_search = GridSearchCV(gb, params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print("Iperparametri ottimali:\n", grid_search.best_params_)
```

In this example, the Gradient Boosting model is defined and the values ​​of the parameters to be tested in the Grid Search, namely learning_rate, n_estimators and max_depth. Then the Grid Search is executed via the GridSearchCV class of sklearn.model_selection, specifying the number of folds cv=5, the number of jobs to run simultaneously n_jobs=-1 and the level of detail of the print verbose=1. Finally, the dictionary of the best parameters found is printed via the best_params_ attribute of the grid_search object.

Once the optimal parameters are found, a new Gradient Boosting model can be created with the found parameters and trained on the data.

```python commands
gb = GradientBoostingClassifier(learning_rate=0.2, max_depth=4, n_estimators=50, random_state=42)

model7 = gb.fit(X_train_scaled, y_train)

prediction7 = model7.predict(X_test_scaled)

cm7 = confusion_matrix(y_test, prediction7)
sns.heatmap(cm7,annot=True)

TP=cm7[0][0]
TN=cm7[1][1]
FN=cm7[1][0]
FP=cm7[0][1]
acc7 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc7)

print(classification_report(y_test, prediction7))
```

In this example, the dataset is split into a training set and a testing set using the train_test_split function of scikit-learn. Next, the Gradient Boosting model is initialized with 100 decision trees of maximum depth 3 and trained on the training set.

Finally, the model performance is evaluated on the testing data using the classification_report function to calculate precision, recall, and F1-score metrics, and the confusion_matrix function to visualize the confusion matrix.

## Naive Bayes Gaussiano

It is a machine learning algorithm that uses Bayes' theorem, assuming feature independence. Although Naïve Bayes guarantees good classification accuracy, the feature independence assumption can often lead to errors in practice. However, due to its computational efficiency and other advantages, Naïve Bayes is widely used.

Here is an example of how to do a Grid Search to optimize the parameters of a Gaussian Naive Bayes model:

A GaussianNB object is created, representing the Gaussian Naive Bayes model. A grid of parameters to test is then defined, in this case priors and var_smoothing. A GridSearchCV object is created, taking as arguments the model, the grid of parameters and the number of folds for cross-validation (in this case cv=5). The model is trained using the grid search, and finally the best parameters found are printed

```python commands
gnb = GaussianNB()

param_grid = {
    'priors': [None, [0.25, 0.75], [0.4, 0.6]],
    'var_smoothing': [1e-9, 1e-7, 1e-5]
}

grid = GridSearchCV(gnb, param_grid, cv=5)

grid.fit(X_train_scaled, y_train)

print("Iperparametri ottimali:\n", grid.best_params_)

gnb = GaussianNB(priors=[0.4, 0.6], var_smoothing=1e-9)

model6 = gnb.fit(X_train_scaled, y_train)

prediction6 = model6.predict(X_test_scaled)

cm6 = confusion_matrix(y_test, prediction6)
sns.heatmap(cm6,annot=True)

TP=cm6[0][0]
TN=cm6[1][1]
FN=cm6[1][0]
FP=cm6[0][1]
acc6 = (TP+TN)/(TP+TN+FN+FP)
print('Testing Accuracy:',acc6)

print(classification_report(y_test, prediction6))
```

In this example, a GaussianNB object is created with the priors parameter equal to [0.4, 0.6] and the var_smoothing parameter set to 1e-9, which represent the optimal values ​​found via the Grid Search. The model is then trained on the training set, and finally the model predictions are compared to the labels in the test set. Finally, the confusion matrix is ​​displayed.

Finally, we calculate the evaluation metrics with accuracy_score, precision_score, recall_score, f1_score.

## Neural Network

An artificial neural network can be a great choice for classification problems like this, where there are many variables that can influence the diagnosis.

```python commands
def create_model(optimizer='adam', activation='relu', neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers = ['adam', 'sgd']
activations = ['relu', 'sigmoid', 'tanh']
neurons = [5, 10, 15, 20]

param_grid = dict(optimizer=optimizers, activation=activations, neurons=neurons)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

grid_result = grid.fit(X_train_scaled, y_train)

print("Iperparametri ottimali:\n", grid_result.best_params_)
```

In this example, we try to find the best hyperparameters for the neural network, choosing the optimizer type (adam or sgd), the activation function for the hidden neurons (relu, sigmoid or tanh) and the number of hidden neurons (5, 10, or 15, 20). The model is built using the Keras library of TensorFlow and trained using the 5-fold cross-validation method. At the end of the GridSearch, the set of optimal hyperparameters found is printed.

```python commands
model = Sequential()
model.add(Dense(20, input_dim=X_train_scaled.shape[1], activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

optimizer = 'adam'
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

test_loss, test_acc = model.evaluate(X_test_scaled, y_test)

y_score_nn = model.predict(X_test_scaled)

training_loss=history.history["loss"]
test_loss=history.history["val_loss"]
epoch_count=range(1,len(training_loss)+1)
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training loss","Test loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
print(history.history.keys())

training_acc=history.history["accuracy"]
test_accu=history.history["val_accuracy"]
plt.plot(epoch_count, training_acc, "r--")
plt.plot(epoch_count, test_accu, "b-")
plt.legend(["Training accuracy","Test accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()

print('Test accuracy:', test_acc)
```

In this example, we use a neural network with 2 hidden layers of 30 and 15 nodes and a ReLU activation function, followed by an output layer with a single node and a sigmoid activation function (since this is a binary classification problem).

We use the Adam optimizer and the binary_crossentropy loss function, typical for binary classification, and evaluate the accuracy of the model during training and on the test set.

## Evaluating models and choosing the best model

For the evaluation of the models we used the ROC (Receiver Operating Characteristic) curve, which is a scheme used to compare the outputs obtained from binary classifiers.

Along the two axes we can represent the recall and (1-(TN/(TN+FP))), respectively represented by True Positive Rate (TPR, fraction of true positives) and False Positive Rate (FPR, fraction of false positives). In other words, we study the ratios between true positive instances (hit rate) and false positives.

```python commands
models = [('Modello Logit', model1), ('Albero decisionale', model2), ('Random Forest', model3), ('KNN', model4), ('SVM', model5), ('Naive Bayes Gaussiano', model6), ('Gradient Boosting', model7)]

fig, ax = plt.subplots()

for name, model in models:
    model.fit(X_train_scaled, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = model.predict(X_test_scaled)

    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)

    roc_auc = auc(fpr, tpr)

    accuracy = model.score(X_test_scaled, y_test)

    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_score_nn, pos_label=1)

roc_auc = auc(fpr_nn, tpr_nn)

plt.plot(fpr, tpr, label=f"Neural Network (AUC = {roc_auc:.2f})", linewidth=2)

plt.legend(loc="lower right")

ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')

plt.show()

x_labels = ['LOG_REG', 'DT','RF', 'KNN', 'SVM', 'GNB', 'GB', 'NN']
y_values = [acc1,acc2,acc3,acc4,acc5,acc6,acc7,test_acc]

acc_modelli = dict(zip(x_labels, y_values))
acc_modelli = dict(sorted(acc_modelli.items(), key=lambda item: item[1], reverse=False))
for a,b in acc_modelli.items():
    plt.text(a, b, str(round(b,3)), fontsize=12, color='dodgerblue', horizontalalignment='right', verticalalignment='bottom')

plt.plot(acc_modelli.keys(), acc_modelli.values(), marker='.', markerfacecolor='dodgerblue', markersize=12, linewidth=4)
plt.xlabel('Modelli')
plt.ylabel('Accuracy')
plt.title('Accuracy dei modelli')
plt.legend(['Modelli'], loc='lower right')
plt.show()

```

## Conclusions

At the end of the presentation, the aim was to provide an overview of machine learning models applied to breast cancer diagnosis.

As we have seen, the use of these techniques can act as a support to the expert eye of the doctor, allowing for a faster and more efficient diagnosis, and thus reducing the risk of false diagnoses, increasing the chances of recovery for patients. However, it is important to be aware of the challenges and limitations of the use of machine learning in the medical field, such as the risk of overfitting.

By continuing to explore these techniques and develop new methodologies, we can pave the way for an even more accurate and personalized diagnosis, further improving the quality of life of patients and the progress of medical research.

# AUTHORS

* Di Siervi Giuseppe 
* Iacuone Diego 
* Kulesko Michele 
* Moriconi Daniele 
* Nticha Sara 
* Chiarilli Cristian


