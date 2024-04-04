from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

def train_model(final_data):
    y = final_data['y']
    X = final_data.drop('y', axis=1)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Build Final model using best obtained parameters from our experiment
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train_resampled, y_train_resampled)
    y_pred = rfc_model.predict(X_test)
    
    # Evaluate model
    print("Confusion Matrics: ", confusion_matrix(y_pred, y_test))
    print("/n Classification Report: ", classification_report(y_test, y_pred))
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    
    # save the model to disk
    filename = 'propensity_model.pkl'
    pickle.dump(rfc_model, open(filename, 'wb'))
    
    print('propensity_model.pkl file dumped in the directory')
    
    return rfc_model

# test a function
if __name__ == '__main__':
    # Read data
    path = "final_version.csv"
    # Load the dataframe
    final_data = pd.read_csv(path)
    train_model(final_data)