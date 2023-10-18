import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_data(CREDITSCORE):
    data = pd.read_csv(CREDITSCORE)
    return data

def preprocess_data(data):
    x = data[["Annual_Income", "Monthly_Inhand_Salary",
              "Num_Bank_Accounts", "Num_Credit_Card",
              "Interest_Rate", "Num_of_Loan",
              "Delay_from_due_date", "Num_of_Delayed_Payment",
              "Credit_Mix", "Outstanding_Debt",
              "Credit_History_Age", "Monthly_Balance"]]
    y = data[["Credit_Score"]]

    x["Credit_Mix"] = x["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 3})

    # Handle missing values or other preprocessing steps if necessary

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return x_scaled, y

def train_model(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

    model = RandomForestClassifier()
    model.fit(xtrain, ytrain.values.ravel())
    
    return model

def predict_credit_score(model):
    a = float(input("Annual Income: "))
    b = float(input("Monthly Inhand Salary: "))
    c = float(input("Number of Bank Accounts: "))
    d = float(input("Number of Credit cards: "))
    e = float(input("Interest rate: "))
    f = float(input("Number of Loans: "))
    g = float(input("Average number of days delayed by the person: "))
    h = float(input("Number of delayed payments: "))
    i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
    j = float(input("Outstanding Debt: "))
    k = float(input("Credit History Age: "))
    l = float(input("Monthly Balance: "))

    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 3}
    i = credit_mix_map.get(i)

    features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
    predicted_score = model.predict(features)
    
    return predicted_score

def main():
    data = load_data("CREDITSCORE.csv")
    x_scaled, y = preprocess_data(data)
    model = train_model(x_scaled, y)
    
    print("Credit Score Prediction :")
    predicted_score = predict_credit_score(model)
    print("Predicted Credit Score =", predicted_score)

if __name__ == "__main__":
    main()
