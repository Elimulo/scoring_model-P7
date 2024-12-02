from evidently.report import Report
from evidently.metrics import DataDriftTable

import pandas as pd

# Load datasets
train_data = pd.read_csv("application_train.csv")
test_data = pd.read_csv("application_test.csv")

# Define Evidently report
report = Report(metrics=[DataDriftTable()])

# Generate report
report.run(reference_data=train_data, current_data=test_data)

# Save as HTML
report.save_html("data_drift_report.html")