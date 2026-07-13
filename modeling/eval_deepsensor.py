import logging
import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# import local modules
script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, '..'])
sys.path.append(os.path.sep.join([project_dir , 'src']))
import eval_model

model_dir = '/gws/pw/j07/aria_giant/wip/draft/acocac/outputs/present/hirham5/4flbw8xy/test_results'
fn = 'snmel_abs_test-no-obs-e1_predictions.nc'
ds = xr.open_zarr(os.path.join(model_dir, f'{fn}.zarr'), chunks='auto')

# run evaluation for all target variables
target_names = ['snmel']
eval_specifier = 'test'

zone_dir = '/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/processed/HIRHAM5-ERAInterim/v_02/GRLzones.zarr'
zones = xr.open_dataset(zone_dir)['zones']
if 'z' in zones.dims:
    zones = zones.set_index(z=['y', 'x']).unstack('z')

for i, target_name in enumerate(target_names):

    m_eval = eval_model.ModelEvaluator(ds, target_name=target_name, batch_size=128, zones=zones)

    logging.info(
        f'{target_name}_true min: {ds[target_name + "_true"].min().compute().item()}, max: {ds[target_name + "_true"].max().compute().item()}.')
    logging.info(
        f'{target_name}_pred min: {ds[target_name + "_pred"].min().compute().item()}, max: {ds[target_name + "_pred"].max().compute().item()}.')

    # calculate scores
    rmse = m_eval.get_rmse()
    mae = m_eval.get_mae()
    mbe = m_eval.get_mbe()
    r2 = m_eval.get_r2()

    if i == 0:  # get main target scores
        rmse_main = rmse
        mae_main = mae
        mbe_main = mbe
        r2_main = r2

    val_fig_dir = os.path.sep.join([model_dir, f'figures'])
    os.makedirs(val_fig_dir, exist_ok=True)

    # plot density of predictions vs target per year
    years = np.unique(ds.time.dt.year.values)
    ax_lims = {'snmel': (-7, 180), 'albedom': None}
    for y in years:
        ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', year=y, ref_line='equal',
                                                x_lims=ax_lims[target_name])
        fig_dir = os.path.sep.join([val_fig_dir, f"{fn}_hexbin.png"])
        ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
        logging.info(f'Saved plot to {fig_dir}.')
        plt.close()

    # # plot density of predictions vs target for each basin separately
    # logging.info(f'Plot groundtruth vs prediction density per basin ...')
    # for basin in basins_nr:
    #     ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', ref_line='equal', zone_cat=basin)
    #     plt.show()
    #     fig_dir = os.path.sep.join([val_fig_dir, f"true_vs_pred_{target_name}_density_basin{str(basin)}.png"])
    #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
    #     logging.info(f'Saved plot to {fig_dir}.')
    #     plt.close()
    #
    # # plot for test year per month
    # year = np.unique(ds.time.dt.year.values)[-1]  # last year in dataset
    # for m in list(calendar.month_name[1:]):
    #     ax = m_eval.plot_pred_vs_target_density(f'{target_name}_true', f'{target_name}_pred', year=year, month=m, ref_line='equal')
    #     fig_dir = os.path.sep.join([val_fig_dir, f"true_vs_pred_{target_name}_density_year{y}_month{m}.png"])
    #     ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
    #     logging.info(f'Saved plot to {fig_dir}.')
    #     plt.close()

    # make maps of predictions, true values, and differences true-pred
    val_fig_dir_map = os.path.sep.join([val_fig_dir, 'maps'])
    os.makedirs(val_fig_dir_map, exist_ok=True)

    # totals per year
    years = np.unique(ds.time.dt.year.values)
    for y in years:
        ax = m_eval.plot_map(year=y, join_colorbar=True)
        plt.show()
        fig_dir = os.path.sep.join([val_fig_dir_map, f"{fn}_map.png"])
        ax.get_figure().savefig(fig_dir, bbox_inches="tight", dpi=300)
        logging.info(f'Saved plot to {fig_dir}.')
        plt.close()