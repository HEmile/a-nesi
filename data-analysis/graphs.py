import pandas
import numpy as np
import matplotlib.pyplot as plt
import sciplot
import torch.distributions

df = pandas.read_csv("measure_speed.csv")
df_dpl = pandas.read_csv("measure_speed_dpl.csv")

class Group:
    def __init__(self, df, predict_only, name):
        self.df = df[df['predict_only'] == predict_only]
        self.name = name

    def query_time(self, N):
        test_times = self.df[self.df["N"] == N]["test_time"]
        avg_tt = np.average(test_times)
        min_tt = np.min(test_times)
        max_tt = np.max(test_times)

        # Normalize by the size of the test set, which is smaller for each N
        divisor = np.floor_divide(10000, 2*N)
        return avg_tt / divisor, min_tt / divisor, max_tt / divisor

    def arrays(self):
        avgs = []
        mins = []
        maxs = []
        for i in range(1, 16):
            tup = self.query_time(i)
            avgs.append(tup[0])
            mins.append(tup[1])
            maxs.append(tup[2])
        return np.array(avgs), np.array(mins), np.array(maxs)

class GroupDPL:
    def __init__(self, method, pretrain, name):
        self.df = df_dpl[np.logical_and(df_dpl['method'] == method, df_dpl['pretrain'] == pretrain)]
        self.method = method
        self.pretrain = pretrain
        self.name = name

    def query_time(self, N):
        test_times = self.df[self.df["N"] == N]
        if len(test_times) == 0:
            return None
        query_times = np.array([test_times[f"run{n}"] for n in range(1, 11)])
        if (query_times == -2).any():
            # Apparently -2 denotes timeout?
            return None
        return np.average(query_times), np.min(query_times), np.max(query_times)

    def arrays(self):
        avgs = []
        mins = []
        maxs = []
        for i in range(1, 16):
            tup = self.query_time(i)
            if tup is not None:
                avgs.append(tup[0])
                mins.append(tup[1])
                maxs.append(tup[2])
        return np.array(avgs), np.array(mins), np.array(maxs)


g_predict = Group(df, True, "predict")
g_joint = Group(df, False, "explain")

sciplot.set_size_cm(9, 5.5)

x = np.arange(1, 16, 1)
x_short = np.arange(1, 6, 1)

def plot_with_sd(subplot, x, avg, min, max, label, color=None, linestyle=None):
    if not subplot:
        subplot = plt
    if not color:
        ax = subplot.plot(x, avg[:len(x)], label=label)
        color = ax[0].get_color()
        linestyle = ax[0].get_linestyle()
    else:
        ax = subplot.plot(x, avg[:len(x)], label=label, color=color, linestyle=linestyle)
    subplot.fill_between(x, min[:len(x)], max[:len(x)], alpha=0.2, color=ax[0].get_color())
    return color, linestyle

# with sciplot.style():
#     for group in dpl_groups:
#         avg, mins, maxs = group.arrays()
#         plot_with_sd(x[:len(avg)], avg, mins, maxs, label=f"{group.name}")
#     plot_with_sd(None, x, *g_joint.arrays(), label="A-NeSI (joint)")
#     plot_with_sd(None, x, *g_predict.arrays(), label="A-NeSI (predict)")
#     plt.xlabel("N")
#     plt.ylabel("Inference time (s)")
#     plt.yscale("log")
#     plt.legend(loc='upper right')
#     # plt.show()
#     plt.savefig("timings.pdf")

def plot_group(group, x, a0, a1, color=None):
    avg, mins, maxs = group.arrays()
    c, l = plot_with_sd(a0, x[:len(avg)], avg, mins, maxs, color=color, label=f"{group.name}")
    a1.plot([], [], label=f"{group.name}", color=c, linestyle=l)
    return c

with sciplot.style():
    f, (a0, a1) = plt.subplots(1, 2, width_ratios=[2, 1])
    plot_group(GroupDPL("exact", 0, "DeepProbLog"), x, a0, a1)

    grounding_dsl = np.array([0.20343, 0.580492, 5.29758286, 58.4800360])
    avg_dsl = np.array([0.002248687, 0.0025327, 0.0040802666, 0.005704576])
    min_dsl = np.array([0.00088119506, 0.0012052059, 0.00340890884399, 0.00402])
    max_dsl = np.array([0.01579976, 0.0068771839, 0.00894212, 0.03940])
    c_dsl, l_dsl = plot_with_sd(a0, np.arange(1, 5, 1), grounding_dsl, grounding_dsl, grounding_dsl, label="DSL (grounding)")

    c_dpla = plot_group(GroupDPL("gm", 0, "DPLA*"), x, a0, a1)
    plot_group(GroupDPL("gm", 128, "DPLA* (pretrain)"), x, a0, a1, c_dpla)
    c1, l1 = plot_with_sd(a0, x, *g_joint.arrays(), label="A-NeSI (explain)")
    c2, l2 = plot_with_sd(a0, x, *g_predict.arrays(), color=c1, label="A-NeSI (predict)")
    plot_with_sd(a1, x, *g_joint.arrays(), label="A-NeSI (joint)", color=c1, linestyle=l1)
    plot_with_sd(a1, x, *g_predict.arrays(), label="A-NeSI (explain)", color=c2, linestyle=l2)

    _, _ = plot_with_sd(a0, np.arange(1, 5, 1), avg_dsl, min_dsl, max_dsl, color=c_dsl, label="DSL (inference)")
    f.tight_layout()
    a0.set_xlabel("N")
    a1.set_xlabel("N")
    a0.set_ylabel("Inference time (s)")
    a0.set_yscale("log")
    a0.legend(loc='upper right')
    # a1.legend(loc='upper left')
    # plt.show()
    plt.savefig("both_timings.pdf")