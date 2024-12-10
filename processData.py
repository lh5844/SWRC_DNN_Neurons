import json
from matplotlib import pyplot as plt
import math

import numpy as np

def readRunData(filepath: str) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val)/(max_val - min_val) if x != min_val else 0.01 for x in data]

# Process Data
run3 = readRunData("runData/3_neurons.json")
run5 = readRunData("runData/5_neurons.json")
run7 = readRunData("runData/7_neurons.json")
run9 = readRunData("runData/9_neurons.json")
run11 = readRunData("runData/11_neurons.json")

rmseData = [run3["RSME"], run5["RSME"], run7["RSME"], run9["RSME"], run11["RSME"]]
d1Data = [run3["d1"], run5["d1"], run7["d1"], run9["d1"], run11["d1"]]
r2Data = [run3["R2"], run5["R2"], run7["R2"], run9["R2"], run11["R2"]]

print(f"~~~~~RUN ANALYSIS:~~~~~~~~~~~~~~\n\t3{'\t'*3}5{'\t'*3}7{'\t'*3}9{'\t'*3}11")
print("d-1:\t"+"\t".join([str(i) for i in d1Data]))
print("RMSE:\t"+"\t".join([str(i) for i in rmseData]))
print("R-Sq.:\t"+"\t".join([str(i) for i in r2Data]))

fig = plt.figure()
ax = plt.subplot()

# Performance Metrics
bar_width = 0.2
cats = ["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"]
x = np.arange(len(cats))
plt.bar(cats, d1Data, width=bar_width, label='d-1')
# plt.bar(x+bar_width, normalize(rmseData), width=bar_width, label='Root Mean-Squared Error')
plt.bar(x-bar_width, r2Data, width=bar_width, label='R-Squared')
plt.title("Performance Metrics For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylim([.98, 1.01])
plt.axhline(1, color="red", linestyle="--", label="Perfect Model")
plt.legend()
plt.ylabel("d-1/R-Sq. Score")
plt.savefig("runData/performance.png")

fig.clear()

# Modified Performance Metrics
bar_width = 0.2
cats = ["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"]
x = np.arange(len(cats))
plt.bar(cats, [abs(1-i) for i in d1Data], width=bar_width, label='d-1')
# plt.bar(x+bar_width, normalize(rmseData), width=bar_width, label='Root Mean-Squared Error')
plt.bar(x-bar_width, [abs(1-i) for i in r2Data], width=bar_width, label='R-Squared')
plt.title("Performance Deviations from Perfection")
plt.xlabel("Neurons per Hidden Layer")
plt.legend()
# plt.ylim([, 1.01])
plt.ylabel("Absolute Difference from 1")
plt.savefig("runData/modified_performance.png")

fig.clear()

# RMSE Comparison
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], normalize(rmseData))
plt.title("Root Mean-Squared Error For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("min-mix Normalized RMSE")
plt.savefig("runData/rmse.png")

# Train time Results
fig = plt.figure()
ax = plt.subplot()
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], [run3["trainTime"], run5["trainTime"], run7["trainTime"], run9["trainTime"], run11["trainTime"]], label="trainTime")
plt.ylim([100, 250])
plt.title("Training Time For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("Training Time (seconds)")
plt.savefig("runData/traintime.png")

fig.clear()

# Convergence
fig = plt.figure()
ax = plt.subplot(111)
i = 3
for run in [run3, run5, run7, run9, run11]:
    v = run["costs"]
    plt.plot([i for i in range(0,5000,20)], [i for i in map(lambda x: math.log(x, 10), v)], label=f"{i} Neurons")
    i += 2
plt.title(f"Convergence Times")
plt.xlabel("Epochs")
plt.legend()
plt.ylabel("log_10(Cost)")
plt.savefig("runData/convergence.png")