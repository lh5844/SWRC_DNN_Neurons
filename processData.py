import json
from matplotlib import pyplot as plt
import math

def readRunData(filepath: str) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

# Process Data
run3 = readRunData("runData/3_neurons.json")
run5 = readRunData("runData/5_neurons.json")
run7 = readRunData("runData/7_neurons.json")
run9 = readRunData("runData/9_neurons.json")
run11 = readRunData("runData/11_neurons.json")

# D-1 Results
fig = plt.figure()
ax = plt.subplot()
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], [run3["d1"], run5["d1"], run7["d1"], run9["d1"], run11["d1"]], label="d1")
plt.ylim([.97, 1.01])
plt.title("d-1 Values For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("d-1")
plt.savefig("d-1.png")

# RMSE Results
fig = plt.figure()
ax = plt.subplot()
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], [run3["RSME"], run5["RSME"], run7["RSME"], run9["RSME"], run11["RSME"]], label="RSME")
plt.ylim([0, .0031])
plt.title("RMSE Values For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("RMSE")
plt.savefig("RMSE.png")

# R2 Results
fig = plt.figure()
ax = plt.subplot()
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], [run3["R2"], run5["R2"], run7["R2"], run9["R2"], run11["R2"]], label="R2")
plt.ylim([.99, .998])
plt.title("R2 Values For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("R2")
plt.savefig("R2.png")

# Train time Results
fig = plt.figure()
ax = plt.subplot()
plt.bar(["3 Neurons","5 Neurons","7 Neurons","9 Neurons","11 Neurons"], [run3["trainTime"], run5["trainTime"], run7["trainTime"], run9["trainTime"], run11["trainTime"]], label="trainTime")
plt.ylim([100, 250])
plt.title("Training Time For Each Run")
plt.xlabel("Neurons per Hidden Layer")
plt.ylabel("Training Time (seconds)")
plt.savefig("traintime.png")

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
plt.savefig("Convergence.png")