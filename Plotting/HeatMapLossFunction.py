import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def HeatMapLossFunction(X_local, y_local, X_global, z_global, f, g, plt, noise_mean, noise_std):
    
    '''
    Plot 2D Heatmap of 3 loss functions:
    local loss, global loss, pertubed global loss

    Assumes a composite function f.g only
    '''
    grid_size = 50
    theta_0_range, theta_1_range = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0.4, 0.7, grid_size))
    local_loss = []
    global_loss = []
    global_pertube_loss = []
    smallest_local = 1e50
    best_local = None
    smallest_global = 1e50
    best_global = None
    with torch.no_grad():
        for theta_0, theta_1 in zip(theta_0_range.flatten(), theta_1_range.flatten()):
            y_pred = g(X_local, params=[theta_0, theta_1], noisy=False)
            z_pred = f(y_pred, noisy = True, noise_mean=noise_mean, noise_std=noise_std)
            #z_pred_pertubed = f(y_pred, noisy = False)
            local_loss_curr = mean_squared_error(y_pred, y_local)
            local_loss.append(local_loss_curr)
            if local_loss_curr < smallest_local:
                smallest_local = local_loss_curr
                best_local = theta_0, theta_1
            
            global_loss_curr = mean_squared_error(z_pred, z_global)
            if global_loss_curr < smallest_global:
                smallest_global = global_loss_curr
                best_global = theta_0, theta_1
                
            global_loss.append(global_loss_curr)
            #global_pertube_loss.append(mean_squared_error(z_pred_pertubed, z_global))

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, remove the last value from the z array.
        local_loss = np.array(local_loss).reshape(theta_0_range.shape)
        z_min, z_max = (local_loss).min(), np.abs(local_loss).max()

        fig, ax = plt.subplots(1,2)

        c = ax[0].pcolormesh(theta_0_range, theta_1_range, local_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax[0].set_title('local loss')
        # set the limits of the plot to the limits of the data
        ax[0].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
        ax[0].scatter([best_local[0]], [best_local[1]], marker='x', s=12,linewidths=4, color="red")
        ax[0].scatter([best_global[0]], [best_global[1]], marker='x', s=12,linewidths=4, color="green")
        #fig.colorbar(c, ax=ax[0])

        global_loss = np.array(global_loss).reshape(theta_0_range.shape)
        c = ax[1].pcolormesh(theta_0_range, theta_1_range, global_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax[1].set_title('system loss')
        # set the limits of the plot to the limits of the data
        ax[1].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
        ax[1].scatter([best_global[0]], [best_global[1]], marker='x', s=12,linewidths=4, color="green", label="system minimum")
        ax[1].scatter([best_local[0]], [best_local[1]], marker='x', s=12,linewidths=4, color="red", label="local minimum")
        #fig.colorbar(c, ax=ax[1])

        # global_pertube_loss = np.array(global_pertube_loss).reshape(theta_0_range.shape)
        # c = ax[2].pcolormesh(theta_0_range, theta_1_range, global_pertube_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
        # ax[2].set_title('pertubed')
        # # set the limits of the plot to the limits of the data
        # ax[2].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
        # fig.colorbar(c, ax=ax[2])
        y_pred = g(X_local, params=[best_global[0], best_global[1]], noisy=False)
        local_loss_found = mean_squared_error(y_pred, y_local)
        theta_0_found = []
        theta_1_found = []
        for theta_0, theta_1 in zip(theta_0_range.flatten(), theta_1_range.flatten()):
            y_pred = g(X_local, params=[theta_0, theta_1], noisy=False)
            local_loss_curr = mean_squared_error(y_pred, y_local)
            if (abs(local_loss_curr - local_loss_found))/local_loss_found < 0.3:
                theta_0_found.append(theta_0)
                theta_1_found.append(theta_1)
        ax[0].scatter(theta_0_found, theta_1_found, marker='s', s=3,linewidths=4, color="gold", label="target region")
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)
        plt.legend(fontsize=10)
        return plt, fig, ax