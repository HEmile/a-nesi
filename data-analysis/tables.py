import math

import pandas
import numpy as np

df = pandas.read_csv("runs.csv")
df_other_methods = pandas.read_csv("results_other_methods.csv", delimiter=';')


df['predict_only'].fillna(False, inplace=True)
df['use_prior'].fillna(True, inplace=True)

Ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

compare_Ns = [1, 2, 4, 15]
ablation_Ns = [1, 3, 6, 10, 15]

class BaseGroup:

    def __init__(self):
        self.should_bold_discretize = {N: False for N in Ns}
        self.should_bold_neural = {N: False for N in Ns}

    def test_accuracy(self, N):
        raise NotImplementedError()

    def amount_runs(self, N):
        return 10

class Group(BaseGroup):

    def __init__(self, df, ablation_name, predict_only, use_prior, prune):
        super().__init__()
        self.df = df[np.logical_and(np.logical_and(
            df['predict_only'] == predict_only,
            df['use_prior'] == use_prior),
            df['prune'] == prune)]
        self.by_N = {d: self.df[self.df["N"] == d] for d in Ns}
        self.ablation_name = ablation_name
        self.should_bold = {N: False for N in Ns}

        for d in Ns:
            assert len(self[d]) == 10

    def __getitem__(self, item):
        return self.by_N[item]

    def test_accuracy(self, N):
        return np.average(self[N]["test_accuracy"]) * 100, np.std(self[N]["test_accuracy"]) * 100

    def test_accuracy_discretize(self, N):
        return np.average(self[N]["test_accuracy_prior"]) * 100, np.std(self[N]["test_accuracy_prior"]) * 100

    def acc_cell(self, N, discretize):
        mean, std = self.test_accuracy_discretize(N) if discretize else self.test_accuracy(N)
        return f"{mean:.2f} $\pm$ {std:.2f}"


class OtherMethodGroup(BaseGroup):

    def __init__(self, df, name, also_has_discretize=False):
        super().__init__()
        self.df = df[df['method'] == name]
        self.also_has_discretize = also_has_discretize
        self.name = name

    def _row(self, N):
        df_for_N = self.df[self.df["N"] == N]
        if len(df_for_N) == 0:
            return None
        return df_for_N.iloc[0]

    def test_accuracy(self, N):
        row = self._row(N)
        if row is None:
            return None
        return row["test_accuracy"], row["test_accuracy_std"]

    def test_accuracy_discretize(self, N):
        row = self._row(N)
        if row is None:
            return None
        return row["test_accuracy_prior"], row["test_accuracy_prior_std"]

    def acc_cell(self, N, discretize):
        tup = self.test_accuracy_discretize(N) if discretize else self.test_accuracy(N)
        if tup is None:
            return "T/O"
        mean, std = tup
        if not std or np.isnan(std):
            return f"{mean:.2f}"
        return f"{mean:.2f} $\pm$ {std:.2f}"

    def amount_runs(self, N):
        return self._row(N)["runs"]


g_full_no_prune = Group(df, "explain", predict_only=False, use_prior=True, prune=False)
g_full_prune = Group(df, "pruning", predict_only=False, use_prior=True, prune=True)
g_predict_prior = Group(df, "predict", predict_only=True, use_prior=True, prune=False)
g_predict_no_prior = Group(df, "no prior", predict_only=True, use_prior=False, prune=False)

g_embed_2_sym = OtherMethodGroup(df_other_methods, "Embed2Sym", True)
g_DPL = OtherMethodGroup(df_other_methods, "DeepProbLog", False)
g_NeurASP = OtherMethodGroup(df_other_methods, "NeurASP", False)
g_DPLAstar = OtherMethodGroup(df_other_methods, "DPLA*", False)
g_DSL = OtherMethodGroup(df_other_methods, "DeepStochLog", False)
g_LTN = OtherMethodGroup(df_other_methods, "LTN", False)
g_NeuPSL = OtherMethodGroup(df_other_methods, "NeuPSL", False)

other_methods = [g_LTN, g_DPL, g_DPLAstar, g_DSL, g_embed_2_sym]

ablation_groups = [g_predict_prior, g_full_no_prune, g_full_prune, g_predict_no_prior]
all_compare_groups = ablation_groups + other_methods

def bold(groups, Ns, discretize):
    for group in groups:
        for N in Ns:
            if discretize:
                group.should_bold_discretize[N] = False
            else:
                group.should_bold_neural[N] = False
    for N in Ns:
        best_group:BaseGroup = None
        best_acc = 0
        best_sdd = 0
        for group in groups:
            if discretize:
                t = group.test_accuracy_discretize(N)
            else:
                t = group.test_accuracy(N)
            if t is None:
                continue
            acc, sdd = t
            if acc > best_acc:
                best_acc = acc
                best_sdd = sdd
                best_group = group
        lower_confidence = best_acc - 1.96 * best_sdd / math.sqrt(best_group.amount_runs(N))
        if discretize:
            best_group.should_bold_discretize[N] = True
        else:
            best_group.should_bold_neural[N] = True
        # for group in groups:
        #     if discretize:
        #         t = group.test_accuracy_discretize(N)
        #     else:
        #         t = group.test_accuracy(N)
        #     if t is None:
        #         continue
        #     acc, sdd = t
        #     upper_confidence = acc + 1.96 * sdd / math.sqrt(group.amount_runs(N))
        #     if upper_confidence > lower_confidence:
        #         if discretize:
        #             group.should_bold_discretize[N] = True
        #         else:
        #             group.should_bold_neural[N] = True

bold(all_compare_groups, compare_Ns, False)
bold(all_compare_groups, compare_Ns, True)

def to_result_row(group, Ns, discretize):
    def to_cell(N):
        bold = group.should_bold_discretize[N] if discretize else group.should_bold_neural[N]
        return f"\\textbf{{{group.acc_cell(N, discretize)}}}" if bold else group.acc_cell(N, discretize)
    return ' & '.join(list(map(to_cell, Ns)))


def print_compare_to_other_methods_table(g_full, g_predict, other_methods, anesi_groups):
    values = ' & '.join(list(map(lambda N: "N=" + str(N), compare_Ns)))
    print(f" & {values}\\\\")
    print("\\hline ")

    print(" & \multicolumn{4}{c}{\\textbf{Symbolic prediction}} \\\\")
    for g in other_methods:
        print(f"{g.name} & {to_result_row(g, compare_Ns, True)}\\\\")
    # print(f"A-NeSI (predict) & {to_result_row(g_predict, compare_Ns, True)}\\\\")
    # print(f"A-NeSI (explain)    & {to_result_row(g_full, compare_Ns, True)}\\\\")
    for group in anesi_groups:
        print("\\textsc{A-NeSI}" + f"({group.ablation_name}) & {to_result_row(group, compare_Ns, True)}\\\\")

    print("\\hline ")
    print(" & \multicolumn{4}{c}{\\textbf{Neural prediction}} \\\\")
    for g in other_methods:
        if g.also_has_discretize:
            print(f"{g.name} & {to_result_row(g, compare_Ns, False)}\\\\")
    # print(f"A-NeSI (predict) & {to_result_row(g_predict, compare_Ns, False)}\\\\")
    # print(f"A-NeSI (explain)    & {to_result_row(g_full, compare_Ns, False)}\\\\")
    for group in anesi_groups:
        print("\\textsc{A-NeSI}" + f"({group.ablation_name}) & {to_result_row(group, compare_Ns, False)}\\\\")
    print("\\hline ")
    print("Reference      &", "            & ".join([f"{100*0.99**(2*N):.2f}" for N in compare_Ns]))

print_compare_to_other_methods_table(g_full_no_prune, g_predict_prior, other_methods, ablation_groups)

for i in range(5):
    print()

def print_compare_ablations(groups):
    values = ' & '.join(list(map(lambda N: "N=" + str(N), ablation_Ns)))
    print(f" & {values}\\\\")
    print("\\hline")

    print("A-NeSI & \multicolumn{5}{c}{\\textbf{Symbolic prediction}} \\\\")
    for group in groups:
        print(f"{group.ablation_name} & {to_result_row(group, ablation_Ns, True)}\\\\")
    print("\hline")
    print("A-NeSI & \multicolumn{5}{c}{\\textbf{Neural prediction}} \\\\")
    for group in groups:
        print(f"{group.ablation_name} & {to_result_row(group, ablation_Ns, False)}\\\\")


bold(ablation_groups, ablation_Ns, False)
bold(ablation_groups, ablation_Ns, True)

print_compare_ablations(ablation_groups)