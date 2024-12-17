import subprocess

# Define the Python executable from the virtual environment
python_executable = "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/env/Scripts/python.exe"

# Run the first script
subprocess.run([python_executable, "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/04_GridSearch CatBoost.py"])

# Run the second script
subprocess.run([python_executable, "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/04_GridSearch XGBoost.py"])