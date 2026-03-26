import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results():
    df = pd.read_csv("results.csv")

    for dataset in df["dataset"].unique():
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df[df["dataset"] == dataset], x="bpp", y="psnr", hue="algorithm", marker="o")
        plt.title(f"PSNR vs. BPP for {dataset} Dataset")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("PSNR (dB)")
        plt.grid(True)
        plt.savefig(f"{dataset}_psnr_vs_bpp.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df[df["dataset"] == dataset], x="bpp", y="ssim", hue="algorithm", marker="o")
        plt.title(f"SSIM vs. BPP for {dataset} Dataset")
        plt.xlabel("Bits Per Pixel (BPP)")
        plt.ylabel("SSIM")
        plt.grid(True)
        plt.savefig(f"{dataset}_ssim_vs_bpp.png")
        plt.close()


if __name__ == "__main__":
    plot_results()
