import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # remove unnecessary columns
    df = df.drop(['month','day'],axis=1)
    
    # Separate categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns
    
    ## Replace unknown values with mode for each categorical variable
    for column in categorical_columns:
        mode_value = df[column].mode()[0]  # Calculate mode
        df[column].replace('unknown', mode_value, inplace=True)  # Replace 'unknown' with mode value
        
    # Apply Label Encoding to categorical variables
    label_encoder = LabelEncoder()
    encoded_categorical = df[categorical_columns].apply(label_encoder.fit_transform)
    
    # Concatenate encoded categorical variables with numerical variables
    encoded_data = pd.concat([encoded_categorical, df[numerical_columns]], axis=1)
    
    # standardize features
    target = encoded_data["y"]
    features = encoded_data.drop("y", axis = 1)
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
        
    # export clean and processed data
    final_data = pd.concat([features, target], axis=1,)
    final_data.to_csv("final_version.csv", index=False)
    print("cleaned data stored in a local directory named as final_version.csv")

    return final_data

# test a function
if __name__ == '__main__':
    # Read data
    path = 'train.csv'
    # Load the dataframe
    df = pd.read_csv(path, sep=';')
    clean_data(df)