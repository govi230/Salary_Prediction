# Import "pandes" library
import pandas

# Store "SalaryData.csv" data into 'dataset' variable
dataset = pandas.read_csv("SalaryData.csv")

# "YearsExperience" is feature for this dataset
X = dataset["YearsExperience"].values.reshape(-1,1)

# "Salary" is target
Y = dataset["Salary"]


from sklearn.linear_model import LinearRegression

# Create Model
model = LinearRegression()

# Train the Model
model.fit(X,Y)

# "dump" method is used to save model into file so that we can use this model further in future
from joblib import dump

# Save model into 'Salary_Predictor' file
dump(model,"Salary_Predictor.pk1")
