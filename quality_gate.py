import json
import sys

MINIMUM_R2 = 0.50

print("Reading model evaluation metrics...")

with open("metrics.json", "r") as file:
    metrics = json.load(file)

r2 = metrics["r2"]

print("Model R2 :", round(r2, 4))
print("Required R2:", MINIMUM_R2)

if r2 < MINIMUM_R2:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
