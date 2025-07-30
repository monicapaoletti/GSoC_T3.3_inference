import os
import datetime

# Define the root directory of your project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))

# Define paths for different sections of the project
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

def results_folder():
    today = datetime.date.today().strftime('%Y-%m-%d') 
    folder = os.path.join(RESULTS_ROOT, today)  
    os.makedirs(folder, exist_ok=True)  
    return folder

