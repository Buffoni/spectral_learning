import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (20, 10)
# reproducibility
random_seed = 42
np.random.seed(random_seed)
# environment
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-civitelli"

"""
alg = {"connectivity": {"elu": ["0.01"],
                        "relu": ["0.01"],
                        "tanh": ["0.01"]},
       "spectral": {"elu": ["0.1"],
                    "relu": ["0.1"],
                    "tanh": ["0.1"]}}
"""
alg = {"connectivity": {"elu": ["0"],
                        "relu": ["0"],
                        "tanh": ["0"]},
       "spectral": {"elu": ["0"],
                    "relu": ["0"],
                    "tanh": ["0"]}}

best_df = []
for df_path in tqdm(glob("./test/*.csv"), "Best curve"):
    dname = df_path.split('_')
    dname[0] = os.path.split(dname[0])[1]
    dname[-1] = dname[-1].split('.')[0]
    df = pd.read_csv(df_path, dtype={"regularizer": "category"})
    best_df.append(df[df["regularizer"].isin(alg[dname[0]][dname[2]])].copy())
    best_df[-1]["method"] = dname[0]
    best_df[-1]["activation"] = dname[2]
    best_df[-1]["type"] = f"{dname[0]}_{dname[2]}"
best_df = pd.concat(best_df)
print(best_df)

plot = sns.lineplot(x="percentile", y="test_accuracy", 
                    hue="activation", style="method",
                    markers=True, dashes=True, 
                    ci="sd", data=best_df)
plot.get_figure().savefig("./test/plot/zero_curve.png")
