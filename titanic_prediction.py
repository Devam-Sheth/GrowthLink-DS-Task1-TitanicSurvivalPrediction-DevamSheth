# Titanic Survival Prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
test_data = pd.read_csv('tested.csv')
print(f"Dataset Shape: {test_data.shape}")
print("\nFirst few rows of the dataset:")
print(test_data.head())

# Extract labels (Survived) from the test data
y = test_data['Survived']

# Since we're using the test data as our entire dataset, let's split it into train and test
X = test_data.drop('Survived', axis=1)

# Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")
print("\nBasic Information:")
print(X.info())

print("\nMissing Values:")
print(X.isnull().sum())

print("\nDescriptive Statistics:")
print(X.describe())

# Visualize survival rate
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=test_data)
plt.title('Survival Distribution')
plt.savefig('survival_distribution.png')
plt.close()

# Survival by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='Survived', data=test_data)
plt.title('Survival by Gender')
plt.savefig('survival_by_gender.png')
plt.close()

# Survival by Passenger Class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=test_data)
plt.title('Survival by Passenger Class')
plt.savefig('survival_by_class.png')
plt.close()

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=test_data, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.savefig('age_distribution.png')
plt.close()

# Correlation matrix
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
correlation_matrix = test_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Data Preprocessing
print("\n--- Data Preprocessing ---")

# Feature Engineering
# Extract titles from names
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print("\nUnique titles:", test_data['Title'].unique())

# Group rare titles
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Lady': 'Royalty',
    'Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Capt': 'Officer',
    'Ms': 'Mrs',
    'Dona': 'Royalty'
}
test_data['Title'] = test_data['Title'].map(title_mapping)

# Create family size feature
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Create binned fare feature
test_data['FareBin'] = pd.qcut(test_data['Fare'], 4, labels=False)

# Extract cabin deck
test_data['Deck'] = test_data['Cabin'].astype(str).str[0]
test_data['Deck'] = test_data['Deck'].replace('n', 'U')  # Replace 'n' (from 'nan') with 'U' for Unknown

# Select features for the model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'Title', 'FamilySize', 'IsAlone', 'Deck']
X = test_data[features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define categorical and numerical features
categorical_features = ['Sex', 'Embarked', 'Title', 'Deck']
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Pclass', 'IsAlone']

# Create preprocessors for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model Selection and Training
print("\n--- Model Training and Evaluation ---")

# Create a pipeline with preprocessing and the classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train and evaluate models
models = {
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gb_pipeline,
    'Logistic Regression': lr_pipeline
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    
    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds))
    
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()
    
    results[name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Feature importance for the best model (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature names after preprocessing
    preprocessor = best_model.named_steps['preprocessor']
    preprocessor.fit(X)
    
    # Get feature names from the preprocessor
    cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + cat_features.tolist()
    
    # Get feature importances from the classifier
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame for easy visualization
    importances_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],  # Ensure lengths match
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importances_df.head(10))
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importances_df.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

# Hyperparameter tuning for the best model
print("\n--- Hyperparameter Tuning ---")

if best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
else:  # Logistic Regression
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }

grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"\nTuned model accuracy on test set: {accuracy_tuned:.4f}")
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))

# Create a Markdown file with the results
results_markdown = f"""# Titanic Survival Prediction Results

## Model Performance Summary
- Best Model: {best_model_name}
- Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}
- Test Precision: {results[best_model_name]['test_precision']:.4f}
- Test Recall: {results[best_model_name]['test_recall']:.4f}
- Test F1 Score: {results[best_model_name]['test_f1']:.4f}

## Tuned Model Performance
- Tuned Test Accuracy: {accuracy_tuned:.4f}
- Best Parameters: {grid_search.best_params_}
"""

with open('results.md', 'w') as f:
    f.write(results_markdown)

print("\nResults saved to 'results.md'")

# Save the final tuned model
import joblib
joblib.dump(tuned_model, 'titanic_survival_model.pkl')
print("Final model saved as 'titanic_survival_model.pkl'")

print("\nAll tasks completed successfully!")
