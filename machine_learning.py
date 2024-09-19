import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Caricamento e preparazione dei dati
# df = pd.read_csv('path_to_your_data.csv')  # Modifica con il percorso del tuo file CSV
# X = df.drop('target', axis=1)  # Sostituisci 'target' con il nome della tua variabile target
# y = df['target']

# Esempio di suddivisione dei dati
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Funzione per creare un modello di rete neurale
def create_model(optimizer='adam', activation='relu', neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
prediction2 = dt.predict(X_test)

print(classification_report(y_test, prediction2))

# Random Forest
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Iperparametri ottimali:\n", grid_search.best_params_)

model3 = RandomForestClassifier(**grid_search.best_params_, random_state=42)
model3.fit(X_train, y_train)
prediction3 = model3.predict(X_test)

cm3 = confusion_matrix(y_test, prediction3)
sns.heatmap(cm3, annot=True)
plt.show()

print(classification_report(y_test, prediction3))

# Gradient Boosting
gb = GradientBoostingClassifier()
params = {'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [2, 3, 4]}
grid_search = GridSearchCV(gb, params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Iperparametri ottimali:\n", grid_search.best_params_)

gb = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
model7 = gb.fit(X_train, y_train)
prediction7 = model7.predict(X_test)

cm7 = confusion_matrix(y_test, prediction7)
sns.heatmap(cm7, annot=True)
plt.show()

print(classification_report(y_test, prediction7))

# Naive Bayes Gaussiano
gnb = GaussianNB()
param_grid = {
    'priors': [None, [0.25, 0.75], [0.4, 0.6]],
    'var_smoothing': [1e-9, 1e-7, 1e-5]
}

grid = GridSearchCV(gnb, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Iperparametri ottimali:\n", grid.best_params_)

gnb = GaussianNB(**grid.best_params_)
model6 = gnb.fit(X_train, y_train)
prediction6 = model6.predict(X_test)

cm6 = confusion_matrix(y_test, prediction6)
sns.heatmap(cm6, annot=True)
plt.show()

print(classification_report(y_test, prediction6))

# Rete Neurale
model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['adam', 'sgd']
activations = ['relu', 'sigmoid', 'tanh']
neurons = [5, 10, 15, 20]

param_grid = dict(optimizer=optimizers, activation=activations, neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X_train, y_train)

print("Iperparametri ottimali:\n", grid_result.best_params_)

model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

optimizer = 'adam'
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

y_score_nn = model.predict(X_test)

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]
epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training loss", "Test loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

training_acc = history.history["accuracy"]
test_accu = history.history["val_accuracy"]
plt.plot(epoch_count, training_acc, "r--")
plt.plot(epoch_count, test_accu, "b-")
plt.legend(["Training accuracy", "Test accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()

print('Test accuracy:', test_acc)

# Valutazione dei modelli e confronto
models = [('Decision Tree', dt), ('Random Forest', model3), ('Gradient Boosting', model7), ('Naive Bayes', model6)]
fig, ax = plt.subplots()

for name, model in models:
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_score_nn, pos_label=1)
roc_auc = auc(fpr_nn, tpr_nn)

plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {roc_auc:.2f})", linewidth=2)
plt.legend(loc="lower right")
ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
plt.show()
