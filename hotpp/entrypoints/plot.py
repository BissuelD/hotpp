import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import sys

def plot_vec(prop):
    """Plots a size-3 vector property (e.g. forces) accuracy"""
    x = np.load("target_{}.npy".format(prop))
    y = np.load("output_{}.npy".format(prop))
    n = np.load("n_atoms.npy")
    while len(n.shape) < len(x.shape):
        n = n[..., None]
    x = x / n
    y = y / n
    fig, ax = plt.subplots(1, 3, figsize=(17, 7))
    fig.subplots_adjust(left=0.05, right=0.95)
    title = ["x", "y", "z"]
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    fig.suptitle("HotPP {0} vs DFT {0}".format(prop), fontsize=16)
    plt.figtext(0.5, 0.01, "Global metrics:    RMSE={:.3f},    MAE={:.3f},    R2={:.3f}".format(rmse, mae, r2), 
                ha='center', fontsize=14, color="teal")
    for i in range(3):
        ax[i].set_aspect(1)
        # title
        ax[i].set_title(title[i] + " component", fontsize=14)
        # axis
        ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
        xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
        ax[i].xaxis.set_major_formatter(xmajorFormatter)
        ax[i].yaxis.set_major_formatter(ymajorFormatter)
        ax[i].set_xlabel('DFT  {}'.format(prop), fontsize=14)
        ax[i].set_ylabel('HotPP {}'.format(prop), fontsize=14)
        ax[i].spines['bottom'].set_linewidth(3)
        ax[i].spines['left'].set_linewidth(3)
        ax[i].spines['right'].set_linewidth(3)
        ax[i].spines['top'].set_linewidth(3)    
        ax[i].tick_params(labelsize=16)
        # scatter
        ax[i].scatter(x[..., i], y[..., i])
        # diagonal line
        s = min(np.min(x[..., i]), np.min(y[..., i]))
        e = max(np.max(x[..., i]), np.max(y[..., i]))
        rmse = np.sqrt(np.mean((x[..., i] - y[..., i]) ** 2))
        mae = np.mean(np.abs(x[..., i] - y[..., i]))
        r2 = 1 - np.sum((x[..., i] - y[..., i]) ** 2) / np.sum((x[..., i] - np.mean(x[..., i])) ** 2)
        ax[i].plot([s, e], [s, e], color='black',linewidth=3,linestyle='--',)
        ax[i].text(0.85 * s + 0.15 * e,
             0.15 * s + 0.85 * e,
             "RMSE: {:.3f}\nMAE: {:.3f}\nR2: {:.3f}".format(rmse, mae, r2), fontsize=14)
    plt.savefig('{}.png'.format(prop))

def plot_matrix(prop):
    """Plots a 3 by 3 matrix property (e.g. polarizability) accuracy"""
    x = np.load("target_{}.npy".format(prop))
    y = np.load("output_{}.npy".format(prop))
    n = np.load("n_atoms.npy")
    while len(n.shape) < len(x.shape):
        n = n[..., None]
    x = x #/ n
    y = y #/ n
    fig, ax = plt.subplots(3, 3, figsize=(20,20))
    fig.subplots_adjust(left=0.05, right=0.95)
    
    title = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    fig.suptitle("HotPP {0} vs DFT {0}".format(prop), fontsize=20)
    plt.figtext(0.5, 0.01, "Global metrics:    RMSE={:.3f},    MAE={:.3f},    R2={:.3f}".format(rmse, mae, r2),
                ha='center', fontsize=18, color="teal")
    for i in range(3):
        for j in range(3):
            ax[i, j].set_aspect(1)
            # title
            ax[i, j].set_title(title[i * 3 + j], fontsize=18)
            # axis
            ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
            xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
            ax[i, j].xaxis.set_major_formatter(xmajorFormatter)
            ax[i, j].yaxis.set_major_formatter(ymajorFormatter)
            ax[i, j].set_xlabel('DFT  {}'.format(prop), fontsize=18)
            ax[i, j].set_ylabel('HotPP {}'.format(prop), fontsize=18)
            ax[i, j].spines['bottom'].set_linewidth(3)
            ax[i, j].spines['left'].set_linewidth(3)
            ax[i, j].spines['right'].set_linewidth(3)
            ax[i, j].spines['top'].set_linewidth(3)    
            ax[i, j].tick_params(labelsize=16)
            # scatter
            ax[i, j].scatter(x[..., i, j], y[..., i, j])
            # diagonal line
            s = min(np.min(x[..., i, j]), np.min(y[..., i, j]))
            e = max(np.max(x[..., i, j]), np.max(y[..., i, j]))
            rmse = np.sqrt(np.mean((x[..., i, j] - y[..., i, j]) ** 2))
            mae = np.mean(np.abs(x[..., i, j] - y[..., i, j]))
            r2 = 1 - np.sum((x[..., i, j] - y[..., i, j]) ** 2) / np.sum((x[..., i, j] - np.mean(x[..., i, j])) ** 2)
            ax[i, j].plot([s, e], [s, e], color='black',linewidth=3,linestyle='--',)
            ax[i, j].text(0.80 * s + 0.20 * e,
                          0.2 * s + 0.8 * e,
                          "RMSE: {:.3f}\nMAE: {:.3f}\nR2: {:.3f}".format(rmse, mae, r2), fontsize=18)
    plt.savefig('{}.png'.format(prop))
    
def plot_tensor_3d(prop):
    """"Plots a 3 by 3 by 3 tensor property (e.g. hyperpolarizability named l3_tensor) accuracy"""
    x = np.load("target_{}.npy".format(prop))
    y = np.load("output_{}.npy".format(prop))
    n = np.load("n_atoms.npy")
    while len(n.shape) < len(x.shape):
        n = n[..., None]
    x = x #/ n
    y = y #/ n
    fig, ax = plt.subplots(3, 9, figsize=(65, 20))
    # smaller margin on the right and left
    fig.subplots_adjust(left=0.05, right=0.95)
    title = [["xxx", "xxy", "xxz", "xyx", "xyy", "xyz", "xzx", "xzy", "xzz"],
             ["yxx", "yxy", "yxz", "yyx", "yyy", "yyz", "yzx", "yzy", "yzz"],
             ["zxx", "zxy", "zxz", "zyx", "zyy", "zyz", "zzx", "zzy", "zzz"]]
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    fig.suptitle("HotPP {0} vs DFT {0}".format(prop), fontsize=30)
    plt.figtext(0.5, 0.01, "Global metrics:    RMSE={:.3f},    MAE={:.3f},    R2={:.3f}".format(rmse, mae, r2),
                ha='center', fontsize=25, color="teal")
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ax[i, j*3 + k].set_aspect(1)
                # title
                ax[i, j*3 + k].set_title(title[i][j*3 + k], fontsize=18)
                # axis
                ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
                xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
                ax[i, j*3 + k].xaxis.set_major_formatter(xmajorFormatter)
                ax[i, j*3 + k].yaxis.set_major_formatter(ymajorFormatter)
                ax[i, j*3 + k].set_xlabel('DFT  {}'.format(prop), fontsize=18)
                ax[i, j*3 + k].set_ylabel('HotPP {}'.format(prop), fontsize=18)
                ax[i, j*3 + k].spines['bottom'].set_linewidth(3)
                ax[i, j*3 + k].spines['left'].set_linewidth(3)
                ax[i, j*3 + k].spines['right'].set_linewidth(3)
                ax[i, j*3 + k].spines['top'].set_linewidth(3)
                ax[i, j*3 + k].tick_params(labelsize=16)
                # scatter
                ax[i, j*3 + k].scatter(x[..., i, j, k], y[..., i, j, k])
                # diagonal line
                s = min(np.min(x[..., i, j, k]), np.min(y[..., i, j, k]))
                e = max(np.max(x[..., i, j, k]), np.max(y[..., i, j, k]))
                rmse = np.sqrt(np.mean((x[..., i, j, k] - y[..., i, j, k]) ** 2))
                mae = np.mean(np.abs(x[..., i, j, k] - y[..., i, j, k]))
                r2 = 1 - np.sum((x[..., i, j, k] - y[..., i, j, k]) ** 2) / np.sum((x[..., i, j, k] - np.mean(x[..., i, j, k])) ** 2)
                ax[i, j*3 + k].plot([s, e], [s, e], color='black',linewidth=3,linestyle='--',)
                ax[i, j*3 + k].text(0.85 * s + 0.05 * e,
                                    0.25 * s + 0.85 * e,
                                    "RMSE: {:.3f}\nMAE: {:.3f}\nR2: {:.3f}".format(rmse, mae, r2),
                                    fontsize=18)
    plt.savefig('{}.png'.format(prop))

def plot_prop(prop):
    if "per_" in prop:
        x = np.load("target_{}.npy".format(prop[4:]))
        y = np.load("output_{}.npy".format(prop[4:]))
        n = np.load("n_atoms.npy")
        while len(n.shape) < len(x.shape):
            n = n[..., None]
        x = x / n
        y = y / n
    else:
        x = np.load("target_{}.npy".format(prop))
        y = np.load("output_{}.npy".format(prop))
    
    # Cusom plot for some properties
    if prop == "dipole" :
        plot_vec(prop)
        return None
    if prop == "polarizability" :
        plot_matrix(prop)
        return None
    if prop == "l3_tensor" :
        plot_tensor_3d(prop)
        return None
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect(1)
    # titile
    plt.title("HotPP {0} vs DFT {0}".format(prop), fontsize=16)
    # axis
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.set_xlabel('DFT  {}'.format(prop), fontsize=14)
    ax.set_ylabel('HotPP {}'.format(prop), fontsize=14)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)    
    ax.tick_params(labelsize=16)
    # scatter
    ax.scatter(x, y)
    # diagonal line
    s = min(np.min(x), np.min(y))
    e = max(np.max(x), np.max(y))
    ax.plot([s, e], [s, e], color='black',linewidth=3,linestyle='--',)
    # rmse
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    print("R2: {:.3f}".format(r2))
    plt.text(0.85 * s + 0.15 * e,
             0.15 * s + 0.85 * e,
             "RMSE: {:.3f}\nMAE: {:.3f}\nR2: {:.3f}".format(rmse, mae, r2), fontsize=14)
    plt.savefig('{}.png'.format(prop))
    print(f"{prop:^12}: {rmse:.4f} {mae:.4f} {r2:.4f}")
    return None


def main(*args, properties=["per_energy", "forces"], **kwargs):
    for prop in properties:
        try:
            plot_prop(prop)
        except:
            print("Fail in plot {}".format(prop))


if __name__ == "__main__":
    main(properties=sys.argv[1:])
