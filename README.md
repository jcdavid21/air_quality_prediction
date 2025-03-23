# Air Quality Prediction App

## Setup Instructions

### 1. Ensure Python 3 is Installed
Check if Python 3 is installed on your system:
```bash
python3 --version
```
If not installed, download it from [python.org](https://www.python.org/downloads/).

### 2. Create a Virtual Environment
#### On macOS/Linux:
```bash
python3 -m venv venv  
source venv/bin/activate  
```

#### On Windows:
```bash
python -m venv venv  
venv\Scripts\activate  
```

### 3. Install Required Dependencies
Run the following command to install necessary libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tkinter folium branca
```

### 4. Run the Application
Execute the following command:
```bash
python3 air_quality_prediction.py
```

## Notes
- Ensure all dependencies are installed correctly.
- Use `deactivate` to exit the virtual environment when done.
- If you encounter issues, try upgrading `pip`:
  ```bash
  pip install --upgrade pip
  ```

