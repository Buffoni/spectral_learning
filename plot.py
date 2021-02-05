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
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-spectral"

for df_path in tqdm(glob("./test/*.csv"), "plot"):
    df = pd.read_csv(df_path, dtype={"regularizer": "category"})

    plt.clf() # clear the current figure
    plot = sns.lineplot(x="percentile", y="test_accuracy", 
                        hue="regularizer", style="regularizer", 
                        markers=True, dashes=False, 
                        ci="sd", data=df)
    plot.get_figure().savefig(os.path.join("./test/plot", os.path.split(df_path.replace(".csv", ".png"))[1]))
