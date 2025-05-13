BASE_PATH = "/home/ivan/PycharmProjects/rbm/mlruns"

skip = [".trash", "0", "models"]

import os
for item in os.listdir(BASE_PATH):
    metric_data = {}
    if item in skip:
        continue
    with open(BASE_PATH + "/" + item + "/meta.yaml", "r") as file:
        for line in file.readlines():
            if line.startswith("name: "):
                print(line)

    a = 2
    for run in os.listdir(BASE_PATH + "/" + item):
        if run == "meta.yaml":
            continue
        with open(BASE_PATH + "/" + item + "/" + run + "/meta.yaml", "r") as file:
            for line in file.readlines():
                if line.startswith("run_name: "):
                    run_name = line
        for metric in os.listdir(BASE_PATH + "/" + item + "/" + run + "/metrics"):
            with open(BASE_PATH + "/" + item + "/" + run + "/metrics" + "/" + metric, "r") as file:
                lines = file.readlines()
                if metric.endswith("f1"):
                    criterion = max
                else:
                    criterion = min
                values = [(float(line.split()[1]), line.split()[2]) for line in lines]
                best_metric, epoch = criterion(values)
                if metric in metric_data:
                    metric_data[metric].append((best_metric, run_name, epoch))
                else:
                    metric_data[metric] = [(best_metric, run_name, epoch)]
                # items.append()
                # print(f"{metric} - {best_metric}")
    if not "train_f1" in list(metric_data.keys()):
        continue
    for key, value in metric_data.items():
        print(key)
        if key == "test_f1":
            data = sorted(value, reverse=True)
        else:
            continue
        for i in data[:5]:
            print(i[0], i[1], i[2])

