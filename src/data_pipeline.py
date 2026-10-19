import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

required_list = ['Patient_ID','Age','Biomarker_Level','Pre_Existing_Conditions','Vaccine_Dose_mcg','Prior_Reaction_History','Reaction_Score']

numeric_columns = ['Age','Biomarker_Level','Vaccine_Dose_mcg']
categorical_columns = 'Prior_Reaction_History'
target = 'Reaction_Score'

def load_dataset(path) :
    data = pd.read_csv(path)
    return data

def validate_columns(data) :
    missing_array = []
    for i in required_list:
        if i not in data.columns:
            missing_array.append(i)
    if missing_array :
        print("Columns are missings")
    else :
        print("no columns are missing")  

def profile_dataset(data):
    print("------------------------------- Dataset Shape --------------------------------------- ")
    print(data.shape)
    print("------------------------------- Dataset Missing values ------------------------------- ")
    print(data.isnull().sum())
    print("------------------------------- Dataset Datatypes ------------------------------------ ")
    print(data.dtypes)
    print("------------------------------- Summary ---------------------------------------------- ")
    print(data.describe())

def cleaning_missing_values(data) :
    Biomarker_Level_median = data["Biomarker_Level"].median()
    data["Biomarker_Level"] = data["Biomarker_Level"].fillna(Biomarker_Level_median)

    Prior_Reaction_History_mode = data["Prior_Reaction_History"].mode()[0]
    data["Prior_Reaction_History"] = data["Prior_Reaction_History"].fillna(Prior_Reaction_History_mode)
    return data

def save_dataset(data) :
    data.to_csv("data/patient_records_clean.csv",index=False)
    print("---------------------------- Clean Dataset Saved Successfully ------------------------")

def prepare_features(data) :
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    encoder = LabelEncoder()
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns])

    x = data[numeric_columns + [categorical_columns  , 'Pre_Existing_Conditions'] ]
    y = data[target]

    return x,y,encoder,scaler

def split_data(x,y) :
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)
    return x_train,x_test,y_train,y_test

def train_logistic_regression(x_train,y_train) :
    model = LogisticRegression()
    model.fit(x_train,y_train)
    return model

def evaluate_logistic_regression(model,x_test,y_test) :
    prediction = model.predict(x_test)
    classifications_report = classification_report(y_test,prediction,output_dict=True) #output_dict=True is written because classification_report return output in string to convert to interger 
    confusions_matrix = confusion_matrix(y_test,prediction)
    return classifications_report,confusions_matrix

def train_random_forest(x_train,y_train) :
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    return model

def compare_best_model(lg_report,rf_report) :
    lg_f1 = lg_report["1"]["f1-score"]
    rf_f1 = rf_report["1"]["f1-score"]
    if lg_f1 > rf_f1 :
        model = "logistic_model"
    else :
        model = "random_forest"
    return model  

def save_best_model (best_model)  :
    joblib.dump(best_model,"outputs/risk_model.joblib")
    print("------------------------ Best model saved Successfully --------------------------------- ")



if __name__ == "__main__" :

    result = load_dataset("data/patient_records.csv")
    validate_columns(result)
    profile_dataset(result)
    cleaned_result = cleaning_missing_values(result)
    save_dataset(cleaned_result)
    print("------------------------------------- checking new data --------------------------------")
    new_data = load_dataset("data/patient_records_clean.csv")
    print("Any null values : ", new_data.isnull().sum())

    x,y,encoder,scaler = prepare_features(new_data)
    print("----------------------------------------- Features --------------------------------------")
    print(x.head())
    print("----------------------------------------- Target ----------------------------------------")
    print(y.head())

    x_train,x_test,y_train,y_test = split_data(x,y)
    print("------------------------------------------ x_train dataset -------------------------------")
    print(x_train.head())
    print("------------------------------------------ y_train dataset -------------------------------")
    print(y_train.head())

    logistic_model = train_logistic_regression(x_train,y_train)
    print(logistic_model)

    logistic_classification_report, logistic_confusion_matrix = evaluate_logistic_regression(logistic_model,x_test,y_test)
    print("--------------------------------- logistic_classification_report --------------------------")
    print(logistic_classification_report)
    print("--------------------------------- logistic_confusion_matrix --------------------------")
    print(logistic_confusion_matrix)

    
    print("--------------------------------- Random Forest classifier ------------------------------- ")
    random_forest_model = train_random_forest(x_train,y_train)
    print(random_forest_model)

    classifications_report1,confusions_matrix1 = evaluate_logistic_regression(random_forest_model,x_test,y_test)
    print("--------------------------------- Random_forest_classification_report --------------------------")
    print(classifications_report1)
    print("--------------------------------- Random_forest_confusion_matrix --------------------------")
    print(confusions_matrix1)

    print("-------------------------------- comaprison of models --------------------------------------- ")
    best_model = compare_best_model(logistic_classification_report,classifications_report1)
    print(best_model)

    if best_model == "logistic_model":
        save_best_model(logistic_model)
    else :
       save_best_model(random_forest_model) 

    
    




















    





    










       











