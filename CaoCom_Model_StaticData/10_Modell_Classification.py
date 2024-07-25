import requests
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def getVitaDBData():
    url = 'https://api.vitaldb.net/cases'
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    else:
        print(f"Failed to fetch the CSV file. Status code: {response.status_code}")
        return None

df = getVitaDBData()

# Data preprocessing
df['sex'] = df['sex'].map({'m': 1, 'f': 0})
columns_to_check = ['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop']
df = df.dropna(subset=columns_to_check)

df['ICU_Class'] = np.where(df['icu_days'] <= 1, 0, 1)

df_majority = df[df['ICU_Class'] == 0]
df_minority = df[df['ICU_Class'] == 1]

df_majority_undersampled = df_majority.sample(len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_undersampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = df_balanced[columns_to_check].values
y_balanced = df_balanced['ICU_Class'].values

X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_balanced_scaled = scaler.transform(X_test_balanced)

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(probability=True, kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'k-Nearest Neighbors': KNeighborsClassifier()
}

def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

results = {}
feature_importances = {}

for clf_name, clf in classifiers.items():
    results[clf_name] = evaluate_classifier(clf, X_train_balanced_scaled, X_test_balanced_scaled, y_train_balanced, y_test_balanced)
    
    if hasattr(clf, 'feature_importances_'):
        feature_importances[clf_name] = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        feature_importances[clf_name] = np.abs(clf.coef_).flatten()

results_df = pd.DataFrame(results).transpose()
print(results_df)

for clf_name, importances in feature_importances.items():
    print(f"\nFeature importances for {clf_name}:")
    for feature, importance in zip(['age', 'sex', 'height', 'weight', 'bmi', 'asa', 'emop'], importances):
        print(f"{feature}: {importance:.4f}")




# Save the model
gb_clf = classifiers['Gradient Boosting']
joblib.dump(gb_clf, 'model_classifier/gradient_boosting_model.pkl')
joblib.dump(scaler, 'model_classifier/scaler.pkl')


# Print mappings
print("Sex mapping: {'m': 1, 'f': 0}")