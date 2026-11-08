import os
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
base_dir = os.path.dirname(os.path.abspath(__file__))
saved = joblib.load(os.path.join(base_dir, "models", "credit_model.pkl"))

model = saved["model"]
income_median = saved["income_median"]
dependents_median = saved["dependents_median"]
columns = saved["columns"]

# thresholds from the "Changing the Threshold"
LOW_LIMIT = 0.3
HIGH_LIMIT = 0.7


def read_number(name, label, low, high, whole=False, required=True):
    # reads one number from the form and checks it
    text = request.form.get(name, "").strip()

    if text == "":
        if required:
            raise ValueError(label + " is required.")
        return None

    try:
        value = float(text)
    except ValueError:
        raise ValueError(label + " must be a number.")

    if not math.isfinite(value):
        raise ValueError(label + " must be a number.")
    if value < low or value > high:
        raise ValueError(label + " must be between " + str(low) + " and " + str(high) + ".")
    if whole and value != int(value):
        raise ValueError(label + " must be a whole number.")

    return value


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            age = read_number("age", "Age", 18, 110, whole=True)
            utilization = read_number("utilization", "Credit utilization", 0, 100000)
            late_30_59 = read_number("late_30_59", "Times 30-59 days late", 0, 50, whole=True)
            late_60_89 = read_number("late_60_89", "Times 60-89 days late", 0, 50, whole=True)
            late_90 = read_number("late_90", "Times 90+ days late", 0, 50, whole=True)
            debt_ratio = read_number("debt_ratio", "Debt ratio", 0, 1000000)
            income = read_number("income", "Monthly income", 0, 10000000, required=False)
            open_lines = read_number("open_lines", "Open credit lines and loans", 0, 100, whole=True)
            real_estate = read_number("real_estate", "Real estate loans", 0, 100, whole=True)
            dependents = read_number("dependents", "Number of dependents", 0, 30, whole=True, required=False)
        except ValueError as e:
            error = str(e)
        else:
            # same steps as in the notebook: flag first, then fill the missing values
            income_missing = 1 if income is None else 0
            if income is None:
                income = income_median
            if dependents is None:
                dependents = dependents_median

            row = {
                "RevolvingUtilizationOfUnsecuredLines": utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": late_90,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfDependents": dependents,
                "late_code_flag": 0,
                "income_missing": income_missing,
                "total_late": late_30_59 + late_60_89 + late_90,
                "over_limit": 1 if utilization > 1 else 0,
            }
            data = pd.DataFrame([row])[columns]
            score = float(model.predict_proba(data)[0, 1])

            if score < LOW_LIMIT:
                level, style = "Low risk", "low"
            elif score < HIGH_LIMIT:
                level, style = "Medium risk", "medium"
            else:
                level, style = "High risk", "high"

            result = {
                "score": round(score, 2),
                "level": level,
                "style": style,
                "total_late": int(row["total_late"]),
                "over_limit": bool(row["over_limit"]),
                "income_missing": bool(income_missing),
            }

    return render_template("index.html", result=result, error=error, form=request.form,
                           low=LOW_LIMIT, high=HIGH_LIMIT)


if __name__ == "__main__":
    app.run(debug=True)