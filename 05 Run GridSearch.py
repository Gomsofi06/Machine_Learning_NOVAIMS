import subprocess

# Define the Python executable from the virtual environment
python_executable = "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/env/Scripts/python.exe" # Change this path to the path of your Python executable

# Controll panel
run_catboost = False
run_xgboost = False
run_random_search = False

if run_random_search:
    subprocess.run([python_executable, "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/04_1 GridSearch RandomSearch.py"])

if run_catboost:
    subprocess.run([python_executable, "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/04_2 GridSearch CatBoost.py"])

if run_xgboost:
    subprocess.run([python_executable, "C:/Users/ruipb/Desktop/Master Projects/ML Project/Machine_Learning_NOVAIMS/04_3 GridSearch XGBoost.py"])

