import matplotlib.pyplot as plt


def viz_time_series(data, labels=None):
    x = []
    y = []
    c = []
    i = 0
    for k, v in data.items():
        for t in v:
            x.append(t)
            y.append(i)
            if labels != None:
                c.append(labels[i])
        i += 1

    if labels != None:
        plt.scatter(x, y, c=c)
    else:
        plt.scatter(x, y)
    plt.xlabel("Normalized Time")
    plt.ylabel("Account Number")
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.show()


def hist_times_across_data(data):
    all_times = []
    for k, v in data.items():
        all_times += v

    plt.hist(all_times)
    plt.xlabel("Normalized Time")
    plt.ylabel("Frequency")
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.show()
