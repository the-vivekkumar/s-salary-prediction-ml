import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset
data = {
    "CGPA": [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.2],
    "Internships": [0, 1, 1, 2, 2, 3, 3],
    "Projects": [1, 2, 2, 3, 4, 4, 5],
    "Certifications": [0, 0, 1, 1, 1, 2, 2],
    "Salary_LPA": [3.5, 4.2, 5.0, 6.5, 7.5, 9.0, 10.0]
}

df = pd.DataFrame(data)

X = df[["CGPA", "Internships", "Projects", "Certifications"]]
y = df["Salary_LPA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 2))

# User input
cgpa = float(input("Enter CGPA: "))
internships = int(input("Enter number of internships: "))
projects = int(input("Enter number of projects: "))
certifications = int(input("Enter number of certifications: "))

# ⭐ FIX: user input as DataFrame
user_data = pd.DataFrame([{
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Certifications": certifications
}])

predicted_salary = model.predict(user_data)[0]

print(f"Predicted Salary: {predicted_salary:.2f} LPA")
