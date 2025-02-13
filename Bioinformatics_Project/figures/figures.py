#%%
import pandas as pd
import seaborn as sns
import os
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
# get current directory
os.getcwd()


#%% 

#PPD analysis of species profiles

data = pd.read_csv("../PPD/train_data/positive_headers_misc.csv")
data["Taxa"] = data["header"].str.split("csv").str[1].str.split("promoters").str[0]

taxa_dict = {'CorynebacteriumglutamicumATCC13032':"Corynebacterium glutamicum", 
             'CampylobacterjejuniRM1221':"Campylobacter jejuni",
                'CorynebacteriumdiphtheriaeNCTC13129':"Corynebacterium diphtheriae",
                'StaphylococcusaureusMW2':"Staphylococcus aureus",
                'EscherichiacolistrK12substrMG1655':"Escherichia coli",
                'ThermococcuskodakarensisKOD1':"Thermococcus kodakarensis",
                'AcinetobacterbaumanniiATCC17978':"Acinetobacter baumannii",
                'PseudomonasputidastrainKT2440':"Pseudomonas putida",
                'StaphylococcusepidermidisATCC12228':"Staphylococcus epidermidis",
                'HaloferaxvolcaniiDS2':"Haloferax volcanii",
                'Campylobacterjejuni81176':"Campylobacter jejuni",
                'StreptococcuspyogenesstrainS119':"Streptococcus pyogenes",
                'BurkholderiacenocepaciaJ2315':"Burkholderia cenocepacia",
                'Campylobacterjejuni81116':"Campylobacter jejuni",
                'Sinorhizobiummeliloti1021':"Sinorhizobium meliloti",
                "BradyrhizobiumjaponicumUSDA110":"Bradyrhizobium japonicum",
                'OnionyellowsphytoplasmaOYM':"Onion yellows phytoplasma",
                'otherspecies':"Other species",
                'NostocspPCC7120':"Nostoc sp",
                'PaenibacillusriograndensisSBR5':"Paenibacillus riograndensis",
                'CampylobacterjejuniNCTC11168':"Campylobacter jejuni",
                'XanthomonascampestrispvcampestrieB100':"Xanthomonas campestris",
                'SynechococcuselongatusPCC7942':"Synechococcus elongatus",
                'Shigellaflexneri5astrM90T':"Shigella flexneri",
                'Helicobacterpyloristrain26695':"Helicobacter pylori",
                'AgrobacteriumtumefaciensstrC58':"Agrobacterium tumefaciens",
                'SynechocystisspPCC6803':"Synechocystis",
                'Bacillussubtilissubspsubtilisstr168':"Bacillus subtilis",
                'KlebsiellaaerogenesKCTC2190':"Klebsiella aerogenes"}

data["Taxa"] = data["Taxa"].map(taxa_dict)
# Calculate the percentage of each taxa
total_count = len(data)
taxa_counts = data["Taxa"].value_counts()
taxa_percentages = (taxa_counts / total_count) * 100

# Create seaborn countplot
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(8, 7))
sns.countplot(data=data, y="Taxa", order=taxa_counts.index, color=".3")

# Add percentage labels at the end of each bar
for i, p in enumerate(ax.patches):
    taxa = p.get_y() + p.get_height() / 2
    percentage = taxa_percentages[taxa_counts.index[i]]
    ax.text(p.get_width() + 10, taxa, f"{percentage:.1f}%", ha="left", va="center")

# ax.set_title("Distribution of species in prokaryotic promoter database")
ax.set_title("")
ax.set_xlabel("Number of promoters")
ax.set_ylabel("")

# change x-axis limit from 0 to 20000
ax.set_xlim(0, 19000)

plt.figure(dpi=1200)
plt.tight_layout()
f.savefig('PPD_Distribution_v2.png', dpi=300, bbox_inches='tight')


#%%


data1 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_10000steps_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data2 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-10000steps-validation_history.csv")[["eval_ACCURACY", "step"]].dropna()

# Create some data to plot
x = list(data1["step"])
y1 = list(data1["eval_ACCURACY"])
y2 = list(data2["eval_ACCURACY"])

y = [y1, y2]  # Store each series of the data in one list

labels = ["10000 steps", "10000/10000 steps"]

# baseline = 0.9

fig, ax = plt.subplots(figsize=(8, 5))

# Define font sizes
SIZE_DEFAULT = 14
SIZE_LARGE = 16
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

baseline = 0.82105

# Plot the baseline
ax.plot(
    [x[0], max(x)],
    [baseline, baseline],
    label="Baseline",
    color="gray",
    linestyle="--",
    linewidth=1,
)

# Plot the baseline text
ax.text(
    x[-1] * 1.01,
    baseline,
    "Baseline",
    color="gray",
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)

# Define a nice color palette:
colors = ["#4E1DD2", "#D21D6A"]

# Plot each of the main lines
for i, label in enumerate(labels):

    if y[i] == 0:
        continue
    else: 
        ax.plot(x, y[i], label=label, color=colors[i], linewidth=2)

        # Text
        ax.text(
            x[-1] * 1.01,
            y[i][-1],
            label,
            color=colors[i],
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

# Hide the all but the bottom spines (axis lines)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(min(x), max(x))

ax.set_xlabel("Number of training steps")
ax.set_ylabel("Validation accuracy")
ax.set_title("Nucleotide Transformer (NT 500m-1000g)")

plt.tight_layout()
plt.savefig("NT_500m_val_acc.png", dpi=800)

#%%


data1 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_10000steps_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data2 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-10000steps-validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data3 = pd.read_csv("../PPD/lora_dropout/D_0.4_MS_10000_r_32_a_1_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()

# Create some data to plot
x = list(data1["step"])
y1 = list(data1["eval_ACCURACY"])
y2 = list(data2["eval_ACCURACY"])
y3 = list(data3["eval_ACCURACY"])

y = [y1, y2, y3]  # Store each series of the data in one list

labels = ["All PPD", "+ Subset PP", "All PPD - LoRA (d=0.4)"]

# baseline = 0.9

fig, ax = plt.subplots(figsize=(8, 5))

# Define font sizes
SIZE_DEFAULT = 14
SIZE_LARGE = 16
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

baseline = 0.82105

# Plot the baseline
ax.plot(
    [x[0], max(x)],
    [baseline, baseline],
    label="Baseline",
    color="gray",
    linestyle="--",
    linewidth=1,
)

# Plot the baseline text
ax.text(
    x[-1] * 1.01,
    baseline,
    "Baseline",
    color="gray",
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)

# Define a nice color palette:
colors = ["#4E1DD2", "#D21D6A", "#0bb4ff"]

# Plot each of the main lines
for i, label in enumerate(labels):

    if y[i] == 0:
        continue
    else: 
        ax.plot(x, y[i], label=label, color=colors[i], linewidth=2)

        # Text
        ax.text(
            x[-1] * 1.01,
            y[i][-1],
            label,
            color=colors[i],
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )

# Hide the all but the bottom spines (axis lines)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(min(x), max(x))

ax.set_xlabel("Number of training steps")
ax.set_ylabel("Validation accuracy")
ax.set_title("Nucleotide Transformer (NT 500m-1000g)")

plt.tight_layout()
plt.savefig("NT_500m_val_acc_v2.png", dpi=800)

#%%

from collections import ChainMap
data1 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_10000steps_test_results.csv").set_index("Unnamed: 0").to_dict()
data1["All PPD"] = data1.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data2 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-10000steps-test_results.csv").set_index("Unnamed: 0").to_dict()
data2["+ Subset PPD"] = data2.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data3 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-1000steps-test_results.csv").set_index("Unnamed: 0").to_dict()
data3["+ Subset PPD (1000 steps)"] = data3.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data4 = pd.read_csv("../PPD/lora_dropout/D_0.4_MS_10000_r_32_a_1_InstaDeepAI_nucleotide-transformer-500m-1000g_test_results.csv").set_index("Unnamed: 0").to_dict()
data4["All PPD - LoRA (d=0.4)"] = data4.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')

data = dict(ChainMap(data1, data2, data3, data4))


colors = ['#961D13', '#DBB289', '#ADC1E2', '#303B5F']

models = list(data.keys())
metrics = ['ACCURACY', 'MCC', 'SENSITIVITY', 'SPECIFICITY']

fig, ax = plt.subplots(figsize=(14, 6))

x = range(len(models))
width = 0.2

for i, metric in enumerate(metrics):
    values = [data[model][f'eval_{metric}'] for model in models]
    ax.bar([j + i*width for j in x], values, width, label=metric, color=colors[i])

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylim(0, 1)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Create a table
table_data = [[f'{data[model][f"eval_{metric}"]:.2f}' for model in models] for metric in metrics]
table = ax.table(cellText=table_data, rowLabels=metrics, colLabels=models, loc='bottom', cellLoc='center')
table.scale(1, 2.5)
table.set_fontsize(25)



plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

plt.savefig("NT_500m_fig1B_v2.png", dpi=800)

#%%

from collections import ChainMap
data1 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_10000steps_test_results.csv").set_index("Unnamed: 0").to_dict()
data1["10000 steps"] = data1.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data2 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-10000steps-test_results.csv").set_index("Unnamed: 0").to_dict()
data2["10000/10000 steps"] = data2.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data3 = pd.read_csv("../PPD/nucleotide-transformer-500m-1000g_finetuned-10000steps-finetuned-1000steps-test_results.csv").set_index("Unnamed: 0").to_dict()
data3["10000/1000 steps"] = data3.pop('InstaDeepAI/nucleotide-transformer-500m-1000g')
data = dict(ChainMap(data1, data2, data3))


colors = ['#961D13', '#DBB289', '#ADC1E2', '#303B5F']

models = list(data.keys())
metrics = ['ACCURACY', 'MCC', 'SENSITIVITY', 'SPECIFICITY']

fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

x = range(len(models))
width = 0.2

for i, metric in enumerate(metrics):
    values = [data[model][f'eval_{metric}'] for model in models]
    ax.bar([j + i*width for j in x], values, width, label=metric, color=colors[i])

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylim(0, 1)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Create a table
table_data = [[f'{data[model][f"eval_{metric}"]:.2f}' for model in models] for metric in metrics]
table = ax.table(cellText=table_data, rowLabels=metrics, colLabels=models, loc='bottom', cellLoc='center')
table.scale(1, 1.5)
table.set_fontsize(10)



plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

# plt.savefig("NT_500m_fig1B.png", dpi=800)

#%%

colors = ["black", "#dc0ab4", "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#b3d4ff", "#00bfa0"]

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


df = pd.read_csv("../PPD/DataScience/NT_fine_tuning_DS_hidden_states.csv")
# Convert Probabilities to integers and into two columns
prob = df[["Sequence", "True Label", "Probabilities"]].copy()

prob["Non-Promoter"] = prob["Probabilities"].apply(lambda x: re.findall(r'[\d\.]+(?:e[+-]?\d+)?', x)[0]).astype(float)
prob["Promoter"] = prob["Probabilities"].apply(lambda x: re.findall(r'[\d\.]+(?:e[+-]?\d+)?', x)[1]).astype(float)
prob = prob.drop("Probabilities", axis=1)
prob["True Label"] = prob["True Label"].astype(int)
predicted_probabilities = prob['Promoter'].astype(float)

# Extract the predicted probabilities for the positive class - promoters
predicted_probabilities = prob['Promoter'].astype(float)

# Generate calibration curve data for the positive class
fraction_of_positives, mean_predicted_value = calibration_curve(
    prob["True Label"], predicted_probabilities, n_bins=10)

# Extract the predicted probabilities for the negative class (non-promoter)
predicted_probabilities_neg = prob['Non-Promoter'].astype(float)

# Generate calibration curve data for the negative class
fraction_of_negatives, mean_predicted_value_neg = calibration_curve(
    1 - prob["True Label"], predicted_probabilities_neg, n_bins=10)

# Create a new figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [2, 1]})

# Plot the calibration curves in the top subplot
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors[1], label="Promoter")
ax1.plot(mean_predicted_value_neg, fraction_of_negatives, "o-", color=colors[-1], label="Non-Promoter")
ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", color=colors[0])

ax1.set_ylabel("Fraction of positives/negatives")
ax1.set_xlabel("")
ax1.set_title("")
ax1.legend(loc="lower right")

# Plot the count subplot in the bottom subplot
counts, bins, _ = ax2.hist(predicted_probabilities, bins=10, histtype='step', lw=2, color=colors[1], label='Promoter')
counts_neg, _, _ = ax2.hist(predicted_probabilities_neg, bins=bins, histtype='step', lw=2, color=colors[-1], label='Non-Promoter')

ax2.set_ylabel("Count")
ax2.set_xlabel("Mean probability scores")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
# plt.savefig("NT_500m_calplot.png", dpi=800)






# %%

colors = ["black", "#dc0ab4", "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#b3d4ff", "#00bfa0"]

import pandas as pd
import h5py
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pacmap
from scipy.spatial.distance import pdist, squareform

all_embeddings = []
all_labels = []

DNA_hash = {"A": 0, "C": 1, "G": 2, "T": 3}
all_sequences = []

# Open the HDF5 file
with h5py.File('../PPD/DataScience/NT_fine_tuning_DS_hidden_states.h5', 'r') as df:
    for i in range(len(df)):
        seq_df = df[str(i)]
        data = seq_df['data'][()]
        label = seq_df['labels'][()]
        probabilities = seq_df['probabilities'][()]
        embeddings = seq_df['embeddings'][()]
        attention = seq_df['attention'][()]
        all_embeddings.append(embeddings)
        all_labels.append(label)
        hashed_data = [DNA_hash[base] for base in data.astype(str)]
        all_sequences.append(np.array(hashed_data))



# initializing the pacmap instance
transformer_embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 

transfromer_transformed = transformer_embedding.fit_transform(all_embeddings, init="pca")

# create a dataframe
df = pd.DataFrame(transfromer_transformed, columns=["x", "y"])
df["labels"] = all_labels
df["labels"] = df["labels"].map({0: "Non-Promoter", 1: "Promoter"})

# create a scatter plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="x", y="y", hue="labels", palette=[colors[1], colors[-1]], alpha=0.75)
# remove legend title
plt.legend(title=None)
# remove axis labels and axis
plt.xlabel("")
plt.ylabel("")
# remove axis outline
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# remove ticks
plt.xticks([])
plt.yticks([])

plt.title("PaCMAP projections of output embeddings")
plt.tight_layout()
plt.savefig("NT_500m_PaCMAP_output.png", dpi=800)

raw_embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
all_sequences_array = np.vstack(all_sequences)
S_transformed = raw_embedding.fit_transform(all_sequences_array, init="pca")

# create a dataframe
df = pd.DataFrame(S_transformed, columns=["x", "y"])
df["labels"] = all_labels
df["labels"] = df["labels"].map({0: "Non-Promoter", 1: "Promoter"})

# create a scatter plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="x", y="y", hue="labels", palette=[colors[1], colors[-1]], alpha=0.75)
# remove legend title
plt.legend(title=None)
# remove axis labels and axis
plt.xlabel("")
plt.ylabel("")
# remove axis outline
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# remove ticks
plt.xticks([])
plt.yticks([])

plt.title("PaCMAP projections of input embeddings")
plt.tight_layout()
plt.savefig("NT_500m_PaCMAP_input.png", dpi=800)


# %%

# Given array of probabilities get the max probability and the associated index
def get_prediction(prob):
    if prob[0] > prob[1]:
        return 0, prob[0]
    else:
        return 1, prob[1]

all_attentions = pd.DataFrame(columns=["label", "prediction", "probability", "attention"])
# Open the HDF5 file
with h5py.File('../PPD/DataScience/NT_fine_tuning_DS_hidden_states.h5', 'r') as df:
    for i in range(len(df)):
        seq_df = df[str(i)]
        data = seq_df['data'][()]
        label = seq_df['labels'][()]
        probabilities = seq_df['probabilities'][()]
        attention = seq_df['attention'][()]
        six_mer = []
        for j, att in enumerate(attention):
            att = att.astype(str)
            if j > 0 and len(att[0]) > 1:
                att = np.append(att, j)
                six_mer.append(att)
        pred_prob =  get_prediction(probabilities)
        new_row = pd.DataFrame({"label": label, "prediction": pred_prob[0], "probability": pred_prob[1], "attention": six_mer}, index=[i]*len(six_mer))
        all_attentions = pd.concat([all_attentions, new_row])


# Get all promoters
promoters = all_attentions[all_attentions["label"] == 1]
# Convert attention column to three new columns
promoters[["k-mer", "attention", "position"]] = pd.DataFrame(promoters["attention"].tolist(), index=promoters.index)
promoters["attention"] = promoters["attention"].astype(float)
promoters["position"] = (promoters["position"].astype(int)*6)-60

# Get all correctly predicted promoters
correct_promoters = promoters[promoters["label"] == promoters["prediction"]]
# Get all incorrectly predicted promoters
incorrect_promoters = promoters[promoters["label"] != promoters["prediction"]]

# Plot the average attention per position for correct and incorrect promoters 

sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=correct_promoters, x="position", y="attention", label="Correctly predicted promoters", color=colors[1], err_style="bars")


# Do the same for non-promoters
non_promoters = all_attentions[all_attentions["label"] == 0]
non_promoters[["k-mer", "attention", "position"]] = pd.DataFrame(non_promoters["attention"].tolist(), index=non_promoters.index)
non_promoters["attention"] = non_promoters["attention"].astype(float)
non_promoters["position"] = (non_promoters["position"].astype(int)*6)-60

# Get all correctly predicted non-promoters
correct_non_promoters = non_promoters[non_promoters["label"] == non_promoters["prediction"]]
# Get all incorrectly predicted non-promoters
incorrect_non_promoters = non_promoters[non_promoters["label"] != non_promoters["prediction"]]
# Plot the average attention per position for correct and incorrect non-promoters
sns.lineplot(data=correct_non_promoters, x="position", y="attention", label="Correctly predicted non-promoters", color=colors[-1], err_style="bars")
# Set axis labels
ax.set_xlabel("Position")
ax.set_ylabel("Average attention weights")
plt.tight_layout()
plt.savefig("NT_500m_attpos.png", dpi=800)


#%%

import logomaker

# Create a salience map for a given sequence
def salience_map(sequence, attention, position):
    map = []
    positions = []
    counts = Counter(sequence)
    pseudo_count = 1E-6
    counts["A"] = attention/(counts["A"]+pseudo_count)
    counts["C"] = attention/(counts["C"]+pseudo_count)
    counts["G"] = attention/(counts["G"]+pseudo_count)
    counts["T"] = attention/(counts["T"]+pseudo_count)

    for i, bp in enumerate(sequence):
        map.append(counts[bp])
        positions = position-i
    return sequence, map, positions

temp = salience_map("ATCGATCGATCG", 0.5, 0)

logomaker.saliency_to_matrix(temp[0], temp[1])


# Get the salience map for given dataframe
def get_salience_map(df):
    salient_maps = pd.DataFrame()
    for i, row in df.iterrows():
        map = salience_map(row["k-mer"], row["attention"], row["position"])
        df = logomaker.saliency_to_matrix(map[0], map[1])
        salient_maps = pd.concat([salient_maps, df], ignore_index=True)

    return salient_maps, df, map

# Get the salience map for correct promoters
nn_df = pd.DataFrame()
for i,row in correct_promoters.groupby(correct_promoters.index):
    salient_map = get_salience_map(row)
    nn_df = nn_df.add(salient_map[0], fill_value=0)

nn_df = nn_df/i


nn_logo = logomaker.Logo(nn_df)

# style using Logo methods
nn_logo.style_spines(visible=False)
nn_logo.style_spines(spines=['left', 'bottom'], visible=True, bounds=[0, .035])

# style using Axes methods
nn_logo.ax.set_xticklabels('%+d'%x for x in [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30])
nn_logo.ax.set_ylabel('Saliency')
nn_logo.ax.set_xlabel('Position')
nn_logo.ax.set_title('Average saliency map for promoters')

plt.tight_layout()
plt.savefig("NT_500m_sal_map_correct_promoters.png", dpi=800)

# Get the salience map for incorrect promoters
nn_df = pd.DataFrame()
for i,row in incorrect_promoters.groupby(incorrect_promoters.index):
    salient_map = get_salience_map(row)
    nn_df = nn_df.add(salient_map[0], fill_value=0)

nn_df = nn_df/i

nn_logo = logomaker.Logo(nn_df)

# style using Logo methods
nn_logo.style_spines(visible=False)
nn_logo.style_spines(spines=['left', 'bottom'], visible=True, bounds=[0, .0065])

# style using Axes methods
nn_logo.ax.set_xticklabels('%+d'%x for x in [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30])
nn_logo.ax.set_ylabel('Saliency')
nn_logo.ax.set_xlabel('Position')
nn_logo.ax.set_title('Average saliency map for misclassified promoters')

plt.tight_layout()
plt.savefig("NT_500m_sal_map_misclass_promoters.png", dpi=800)

# Get the salience map for correct non-promoters
nn_df = pd.DataFrame()
for i,row in correct_non_promoters.groupby(correct_non_promoters.index):
    salient_map = get_salience_map(row)
    nn_df = nn_df.add(salient_map[0], fill_value=0)

nn_df = nn_df/i

nn_logo = logomaker.Logo(nn_df)

# style using Logo methods
nn_logo.style_spines(visible=False)
nn_logo.style_spines(spines=['left', 'bottom'], visible=True, bounds=[0, .0065])

# style using Axes methods
nn_logo.ax.set_xticklabels('%+d'%x for x in [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30])
nn_logo.ax.set_ylabel('Saliency')
nn_logo.ax.set_xlabel('Position')
nn_logo.ax.set_title('Average saliency map for non-promoters')

plt.tight_layout()
plt.savefig("NT_500m_sal_map_nonpromoters.png", dpi=800)

# Get the salience map for incorrect non-promoters
nn_df = pd.DataFrame()
for i,row in incorrect_non_promoters.groupby(incorrect_non_promoters.index):
    salient_map = get_salience_map(row)
    nn_df = nn_df.add(salient_map[0], fill_value=0)

nn_df = nn_df/i

nn_logo = logomaker.Logo(nn_df)

# style using Logo methods
nn_logo.style_spines(visible=False)
nn_logo.style_spines(spines=['left', 'bottom'], visible=True, bounds=[0, .0065])

# style using Axes methods
nn_logo.ax.set_xticklabels('%+d'%x for x in [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30])
nn_logo.ax.set_ylabel('Saliency')
nn_logo.ax.set_xlabel('Position')
nn_logo.ax.set_title('Average saliency map for misclassified non-promoters')

plt.tight_layout()
plt.savefig("NT_500m_sal_map_misclass_nonpromoters.png", dpi=800)

# %%
# Get k-mer composition and other properties per correct_promoters and incorrect_promoters
from collections import Counter
correct_promoters["k-mer"] = correct_promoters["k-mer"].astype(str)
incorrect_promoters["k-mer"] = incorrect_promoters["k-mer"].astype(str)

# Get the k-mer from correct promoters that has the max attention weight
correct_promoters_kmers = set(correct_promoters.groupby(correct_promoters.index)["k-mer"].transform("max"))
# Get the k-mer from incorrect promoters that has the max attention weight
incorrect_promoters_kmers = set(incorrect_promoters.groupby(incorrect_promoters.index)["k-mer"].transform("max"))

# Get the k-mer from incorrect promoters that has the max attention weight
incorrect_promoters_kmers = set(incorrect_promoters.groupby(incorrect_promoters.index)["k-mer"].transform("max"))
# Get the k-mer from correct promoters that has the max attention weight
correct_promoters_kmers = set(correct_promoters.groupby(correct_promoters.index)["k-mer"].transform("max"))

# Get the k-mer composition for correct promoters
correct_promoters_kmers = Counter(correct_promoters["k-mer"].values)
# Get the k-mer composition for incorrect promoters
incorrect_promoters_kmers = Counter(incorrect_promoters["k-mer"].values)

# Get the k-mer composition for correct non-promoters
correct_non_promoters_kmers = Counter(correct_non_promoters["k-mer"].values)
# Get the k-mer composition for incorrect non-promoters
incorrect_non_promoters_kmers = Counter(incorrect_non_promoters["k-mer"].values)




# %%


#%%


data1 = pd.read_csv("../PPD/lora_dropout/D_0.1_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data2 = pd.read_csv("../PPD/lora_dropout/D_0.2_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data3 = pd.read_csv("../PPD/lora_dropout/D_0.3_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data4 = pd.read_csv("../PPD/lora_dropout/D_0.4_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data5 = pd.read_csv("../PPD/lora_dropout/D_0.5_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data6 = pd.read_csv("../PPD/lora_dropout/D_0.6_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data7 = pd.read_csv("../PPD/lora_dropout/D_0.7_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data8 = pd.read_csv("../PPD/lora_dropout/D_0.8_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()
data9 = pd.read_csv("../PPD/lora_dropout/D_0.9_InstaDeepAI_nucleotide-transformer-500m-1000g_validation_history.csv")[["eval_ACCURACY", "step"]].dropna()

# Create some data to plot
x = list(data1["step"])
y1 = list(data1["eval_ACCURACY"])
y2 = list(data2["eval_ACCURACY"])
y3 = list(data3["eval_ACCURACY"])
y4 = list(data4["eval_ACCURACY"])
y5 = list(data5["eval_ACCURACY"])
y6 = list(data6["eval_ACCURACY"])
y7 = list(data7["eval_ACCURACY"])
y8 = list(data8["eval_ACCURACY"])
y9 = list(data9["eval_ACCURACY"])


y = [y1, y2, y3, y4, y5, y6, y7, y8, y9]

labels = ["d=0.1", "d=0.2", "d=0.3", "d=0.4", "d=0.5", "d=0.6", "d=0.7", "d=0.8", "d=0.9"]

# baseline = 0.9

fig, ax = plt.subplots(figsize=(8, 5))

# Define font sizes
SIZE_DEFAULT = 14
SIZE_LARGE = 16
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

# baseline = 0.82105

# Plot the baseline
# ax.plot(
#     [x[0], max(x)],
#     [baseline, baseline],
#     label="Baseline",
#     color="gray",
#     linestyle="--",
#     linewidth=1,
# )

# Plot the baseline text
# ax.text(
#     x[-1] * 1.01,
#     baseline,
#     "Baseline",
#     color="gray",
#     fontweight="bold",
#     horizontalalignment="left",
#     verticalalignment="center",
# )

# Define a nice color palette:
# colors = ["#4E1DD2", "#D21D6A", "#0bb4ff"]
colors = ["black", "#e60049", "#e6d800", "#0bb4ff", "#50e991",  "#9b19f5", "#ffa300", "#b3d4ff", "#00bfa0"]


# Plot each of the main lines
for i, label in enumerate(labels):

    if y[i] == 0:
        continue
    if i == 3: 
        ax.plot(x, y[i], label=label, color=colors[i], linewidth=2.5)
        # add legend
        ax.legend(bbox_to_anchor=(0.95, 0.6), ncol=1)
    else:
        ax.plot(x, y[i], label=label, color=colors[i], linewidth=1, linestyle="--",)
        # add legend
        ax.legend(bbox_to_anchor=(0.95, 0.6), ncol=1)
    

        # # Text
        # ax.text(
        #     x[-1] * 1.01,
        #     y[i][-1],
        #     label,
        #     color=colors[i],
        #     fontweight="bold",
        #     horizontalalignment="left",
        #     verticalalignment="center",
        # )

# Hide the all but the bottom spines (axis lines)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(min(x), max(x))

ax.set_xlabel("Number of training steps")
ax.set_ylabel("Validation accuracy")
ax.set_title("Nucleotide Transformer (NT 500m-1000g)")

plt.tight_layout()
plt.savefig("NT_500m_lora_d_test.png", dpi=800)
# %%
