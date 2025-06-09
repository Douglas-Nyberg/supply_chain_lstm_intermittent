#////////////////////////////////////////////////////////////////////////////////#
# File:         gen_mc_data.py                                                   #
# Author:       Douglas Nyberg, logic derived from Noah                          #
# Email:        douglas1.nyberg@gmail.com                                        #
# Date:         2025-06-06                                                       #
# Description:  Generate synthetic monte carlo demand data for forecasting.      #
# Affiliation:  Physics Department, Purdue University                            #
#////////////////////////////////////////////////////////////////////////////////#

"""
Generate synthetic monte carlo demand data for forecasting experiments.

This generates intermittent demand patterns using spike-based simulation.
Basically we randomly place demand spikes in time and add some falloff.
"""

import argparse
import math
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import scipy.stats as stats

# add project root so we can import stuff
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# import mc config
from experiment_workflows.exp_per_item_store_mc import config_mc_master


# truncated normal sampler class
# this lets us sample from a normal distribution but constrained to a range
class TruncNormSampling:
    def __init__(self, data_low: float, data_high: float, center: float, std: float):
        # prevent division by zero
        if std <= 1e-6:
            std = 1e-6  # tiny but nonzero
        
        # standardize bounds for truncnorm
        self.a = (data_low - center) / std
        self.b = (data_high - center) / std
        self.dist = stats.truncnorm(a=self.a, b=self.b, loc=center, scale=std)

    def sample(self, amount: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """sample from the truncated normal distribution"""
        if rng is not None:
            return self.dist.rvs(size=amount, random_state=rng)
        else:
            # fallback but not great for reproducability
            return self.dist.rvs(size=amount)


class SECEasing:
    """sech squared easing function for demand falloff around spikes
    
    this creates a nice smooth falloff pattern around demand peaks
    kinda like a bell curve but with sech squared which looks better
    """
    def __init__(self, center: float = 0, amplitude: float = 1, steepness: float = 1):
        self.amplitude = amplitude
        self.center = center
        self.steepness = max(steepness, 1e-6)  # dont want zero steepness

    def get_image(self, pos: float) -> float:
        """get the falloff value at position pos"""
        if self.amplitude == 0:  # quick exit if no amplitude
            return 0.0
        
        cosh_val = np.cosh(self.steepness * (pos - self.center))
        if np.isinf(cosh_val):  # prevent overflow
             return 0.0
        
        # sech squared formula: sech(x)^2 = (1/cosh(x))^2
        return self.amplitude * math.pow(1 / cosh_val, 2)


# demand spike generation functions
def match_amplitude_spikes(
    time_delta_sampling_func: Callable[..., np.ndarray],  # function that samples time deltas
    time_frame: int,
    rng: np.random.Generator
) -> list[int]:
    """figure out when demand spikes should occur
    
    we sample random time deltas between spikes and place them throughout
    the time frame. this gives us intermittent demand patterns.
    """
    total_time = 0
    spike_indices = []  # when spikes happen
    
    while True:
        # sample how many days until next spike
        delta = math.floor(time_delta_sampling_func(amount=1, rng=rng)[0])
        delta = max(1, delta)  # make sure time advances
        
        total_time += delta
        if total_time >= time_frame:
            break  # past end of timeframe
        else:
            spike_indices.append(total_time)
    
    return spike_indices


def sample_demand(
    time_delta_sampling_func: Callable[..., np.ndarray],
    amplitude_sampling_func: Callable[..., np.ndarray],
    falloff_steepness_sampling_func: Callable[..., np.ndarray],
    time_frame: int,
    max_ease_range: int,
    rng: np.random.Generator
) -> np.ndarray:
    """generate a full demand time series
    
    this is the main function that creates intermittent demand.
    we place spikes randomly in time, then add falloff around each spike.
    """
    
    # start with zeros everywhere
    demand_series = np.zeros(time_frame, dtype=float)
    
    # figure out when spikes happen
    spike_indices = match_amplitude_spikes(time_delta_sampling_func, time_frame, rng)

    # add each spike and its falloff
    for spike_idx in spike_indices:
        if 0 <= spike_idx < time_frame:
            # sample how big this spike should be
            peak_amplitude = math.floor(amplitude_sampling_func(amount=1, rng=rng)[0])
            if peak_amplitude <= 0:
                continue  # skip zero spikes
            
            # add the main spike
            demand_series[spike_idx] += peak_amplitude

            # create falloff around the spike
            steepness = falloff_steepness_sampling_func(amount=1, rng=rng)[0]
            easing_func = SECEasing(spike_idx, peak_amplitude, steepness)
            
            # figure out window around spike to apply falloff
            window_start = max(0, spike_idx - max_ease_range)
            window_end = min(time_frame, spike_idx + max_ease_range + 1)

            # apply falloff to nearby time points
            for t in range(window_start, window_end):
                if t == spike_idx:
                    continue  # already added main spike
                demand_series[t] += easing_func.get_image(float(t))
                
    return demand_series


# main data generation function
def generate_and_save_data(config_profile: str, output_hdf5_path: str, random_seed: int = None):
    """generate synthetic monte carlo demand data and save to hdf5
    
    this is the main orchestration function that generates all the data
    we need for the experiment. creates intermittent demand patterns.
    """
    print(f"loading config from config_mc_master.py")
    # TODO: maybe use different profiles later? for now just use default
    mc_config = config_mc_master.MC_DATA_CONFIG
    
    if not mc_config:
        print("couldnt load mc config, something is wrong")
        return

    # setup random number generator
    rng = np.random.default_rng(random_seed)
    print(f"setup rng with seed: {random_seed if random_seed is not None else 'None'}")

    # get basic config params
    num_items = mc_config.get('num_items', 1)
    num_years = mc_config.get('num_years', 1)
    time_frame_days = num_years * 365  # assume 365 days per year
    
    print(f"generating data for: {num_items} items, {num_years} years ({time_frame_days} days total)")

    # generate static per-item data like costs and lead times
    print("creating static data (inventory levels, costs, lead times etc)...")
    
    # initial inventory levels
    initial_serviceable = rng.integers(
        0, mc_config.get('serviceable_upper_limit', 20) + 1,
        size=(1, num_items)
    )
    initial_unserviceable = rng.integers(
        0, mc_config.get('unserviceable_upper_limit', 10) + 1,
        size=(1, num_items)
    )
    
    # cost data
    procurement_costs = rng.random(size=num_items) * mc_config.get('procurement_cost_upper_limit', 100)
    repair_costs = rng.random(size=num_items) * mc_config.get('repair_cost_upper_limit', 100)
    
    # lead time data
    procurement_lead_times = rng.integers(
        1, mc_config.get('procurement_lead_time_upper_limit', 20) + 1,
        size=num_items
    )
    repair_lead_times = rng.integers(
        1, mc_config.get('repair_lead_time_upper_limit', 20) + 1,
        size=num_items
    )

    # setup containers for demand data
    all_daily_demand = {}  # store daily demand for each item
    all_quarterly_demand = {}  # quarterly aggregates
    all_biannual_demand = {}  # biannual aggregates  
    all_yearly_demand = {}  # yearly aggregates
    all_demand_parameters = {}  # parameters used for each item

    # get demand generation parameters from config
    height_range_cfg = mc_config.get('height_range', {"LOW": 2, "HIGH": 20})
    height_granularity = mc_config.get('height_granularity', 1)
    height_means_options = np.arange(
        height_range_cfg['LOW'], 
        height_range_cfg['HIGH'] + height_granularity, 
        height_granularity
    )
    
    delta_time_range_cfg = mc_config.get('delta_time_range', {"LOW": 10, "HIGH": 100})
    delta_time_granularity = mc_config.get('delta_time_granularity', 2)
    delta_time_means_options = np.arange(
        delta_time_range_cfg['LOW'], 
        delta_time_range_cfg['HIGH'] + delta_time_granularity, 
        delta_time_granularity
    )

    max_ease_range = mc_config.get('max_ease_range', 1)  # falloff range around spikes

    # generate demand data for each item
    print(f"starting demand generation for {num_items} items over {num_years} years...")
    for i in range(num_items):
        print(f"  working on item {i+1}/{num_items}...")
        
        # pick random parameters for this item's demand pattern
        chosen_height_mean = rng.choice(height_means_options)
        chosen_delta_time_mean = rng.choice(delta_time_means_options)

        # setup height (amplitude) parameters
        h_mean = chosen_height_mean
        h_std = max(1.0, h_mean / 3.0)  # std dev as fraction of mean
        h_low = 0.0  # cant have negative demand
        h_high = max(h_mean * 2, height_range_cfg['HIGH'])  # upper bound
        height_params = [h_low, h_high, h_mean, h_std]

        # setup time delta parameters (days between spikes)
        dt_mean = chosen_delta_time_mean
        dt_std = max(1.0, dt_mean / 3.0)  # std dev
        dt_low = max(1.0, delta_time_range_cfg['LOW'] / 2.0)  # at least 1 day
        dt_high = max(dt_mean * 2, delta_time_range_cfg['HIGH'])  # upper bound
        delta_time_params = [dt_low, dt_high, dt_mean, dt_std]
        
        # steepness parameters for falloff shape
        s_mean, s_std, s_low, s_high = 0.2, 0.1, 0.01, 1.0  # reasonable defaults
        steepness_params = [s_low, s_high, s_mean, s_std]

        # save parameters for this item
        all_demand_parameters[f'demand_parameters_item_{i}'] = [delta_time_params, height_params, steepness_params]

        # create samplers for this item using the parameters
        height_sampler = TruncNormSampling(*height_params)
        delta_time_sampler = TruncNormSampling(*delta_time_params)
        steepness_sampler = TruncNormSampling(*steepness_params)

        # generate the actual daily demand time series
        daily_demand_float = sample_demand(
            delta_time_sampler.sample, 
            height_sampler.sample,
            steepness_sampler.sample,
            time_frame_days,
            max_ease_range,
            rng
        )
        
        # convert to integers and make sure no negative demand
        daily_demand_int = np.round(daily_demand_float).astype(int)
        daily_demand_int[daily_demand_int < 0] = 0  # demand cant be negative
        all_daily_demand[f'daily_demand_item_{i}'] = daily_demand_int.reshape(1, -1)

        # aggregate daily demand into different time periods
        yearly_totals = []
        quarterly_totals = []
        biannual_totals = []

        # process each year separately
        for year_idx in range(num_years):
            year_start_idx = year_idx * 365
            year_data = daily_demand_int[year_start_idx : year_start_idx + 365]
            
            # yearly total is just sum of all days
            yearly_totals.append(np.sum(year_data))
            
            # split year into quarters (roughly 90 days each)
            # this is approximate since 365/4 = 91.25
            q1_sum = np.sum(year_data[0:90])
            q2_sum = np.sum(year_data[90:181])  # 91 days
            q3_sum = np.sum(year_data[181:273])  # 92 days
            q4_sum = np.sum(year_data[273:365])  # 92 days
            quarterly_totals.extend([q1_sum, q2_sum, q3_sum, q4_sum])

            # split year into halves
            h1_sum = np.sum(year_data[0:182])  # first half (182 days)
            h2_sum = np.sum(year_data[182:365])  # second half (183 days)
            biannual_totals.extend([h1_sum, h2_sum])
        
        # save aggregated data
        all_yearly_demand[f'yearly_demand_item_{i}'] = np.array(yearly_totals).reshape(1, -1)
        all_quarterly_demand[f'quarterly_demand_item_{i}'] = np.array(quarterly_totals).reshape(1, -1)
        all_biannual_demand[f'biannual_demand_item_{i}'] = np.array(biannual_totals).reshape(1, -1)

    # save everything to hdf5 file
    print(f"saving all data to: {output_hdf5_path}")
    with h5py.File(output_hdf5_path, 'w') as f:
        # save demand time series at different aggregation levels
        print("  saving daily demand data...")
        for key, value in all_daily_demand.items():
            f.create_dataset(key, data=value, compression="gzip")
        
        print("  saving quarterly demand data...")
        for key, value in all_quarterly_demand.items():
            f.create_dataset(key, data=value, compression="gzip")
        
        print("  saving biannual demand data...")
        for key, value in all_biannual_demand.items():
            f.create_dataset(key, data=value, compression="gzip")
        
        print("  saving yearly demand data...")
        for key, value in all_yearly_demand.items():
            f.create_dataset(key, data=value, compression="gzip")
        
        # save the parameters used to generate each item's demand
        print("  saving demand generation parameters...")
        for key, value in all_demand_parameters.items():
            f.create_dataset(key, data=np.array(value), compression="gzip")

        # save static item data
        print("  saving static item data...")
        f.create_dataset("per_item_serviceable_amounts", data=initial_serviceable, compression="gzip")
        f.create_dataset("per_item_unserviceable_amounts", data=initial_unserviceable, compression="gzip")
        f.create_dataset("rand_procurement_cost_per_item", data=np.round(procurement_costs, decimals=2), compression="gzip")
        f.create_dataset("rand_repair_cost_per_item", data=np.round(repair_costs, decimals=2), compression="gzip")
        f.create_dataset("rand_procurement_lead_time_per_item", data=procurement_lead_times, compression="gzip")
        f.create_dataset("rand_repair_lead_time_per_item", data=repair_lead_times, compression="gzip")
    
    print("data generation finished successfully!")


# main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate synthetic demand data using monte carlo simulation")
    parser.add_argument(
        "-p", "--profile",
        default="default",
        help="config profile to use (not implemented yet but planned)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="path to output hdf5 file where data will be saved (optional, uses config default)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="random seed for reproducability (optional)"
    )
    args = parser.parse_args()

    # use default output path if none provided
    if args.output is None:
        args.output = config_mc_master.MC_DATA_CONFIG.get("default_output_file")
        if args.output is None:
            print("ERROR: No output path provided and no default in config")
            sys.exit(1)
        print(f"Using default output path from config: {args.output}")

    # make sure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"created output directory: {output_dir}")

    # run the data generation
    generate_and_save_data(args.profile, args.output, args.seed)