# solveit-calculator
a math automation tool for engineering

# SolveIt v5

SolveIt v5 is a Python web-based calculator with three main modes:

- Scientific calculator
- Polynomial solver
- Function analyzer with graphing

## Run locally

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open your browser at:
   ```text
   http://127.0.0.1:5000
   ```

## Notes

- The app uses Flask, SymPy, NumPy, and Plotly.
- Logs are stored in the `logs/` directory.
