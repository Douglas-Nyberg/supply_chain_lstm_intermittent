#////////////////////////////////////////////////////////////////////////////////#
# File:         convert_hdf5_to_m5.py                                           #
# Author:       Douglas Nyberg                                                  #
# Email:        douglas1.nyberg@gmail.com                                       #
# Date:         2025-06-04                                                      #
#////////////////////////////////////////////////////////////////////////////////#
#!/usr/bin/env python3
"""
HDF5 to Simplified M5 Format Converter

Converts synthetic demand data from gen_mc_data.py HDF5 format to M5-compatible
CSV structure with simplified features (no events, SNAP, or complex pricing).

Output files:
- calendar.csv: Basic date features only
- sales_train_validation.csv: Synthetic demand mapped to M5 item structure  
- sell_prices.csv: Optional placeholder prices (constant values)
"""

import argparse
import h5py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simplified_calendar(start_date: str, number_of_days: int) -> pd.DataFrame:
    """
    Create simplified calendar.csv with basic date features only.
    
    Excludes M5-specific features like events, SNAP, holidays.
    """
    logger.info(f"Creating simplified calendar for {number_of_days} days starting {start_date}")
    
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=number_of_days, freq='D')
    
    calendar_data = []
    for i, date in enumerate(dates):
        # Basic date features only - no events, SNAP, holidays
        calendar_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'd': f'd_{i+1}',  # M5 day index format
            'wm_yr_wk': f"{date.year}_{date.isocalendar()[1]:02d}",  # Year_Week
            'weekday': date.day_name(),
            'wday': date.weekday() + 1,  # 1-7 (Monday=1)
            'month': date.month,
            'year': date.year,
            # Placeholder for M5 structure but not used
            'event_name_1': None,
            'event_type_1': None,
            'event_name_2': None, 
            'event_type_2': None,
            'snap_CA': 0,
            'snap_TX': 0,
            'snap_WI': 0
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    logger.info(f"Created calendar with {len(calendar_df)} rows")
    return calendar_df

def create_sales_data_from_hdf5(hdf5_path: str, num_items: int, num_days: int) -> pd.DataFrame:
    """
    Convert HDF5 daily demand data to M5 sales_train_validation.csv format.
    
    Maps synthetic items to M5-style hierarchical structure.
    """
    logger.info(f"Converting HDF5 demand data to M5 sales format")
    
    sales_data = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(num_items):
            # Get daily demand for this item
            demand_key = f'daily_demand_item_{i}'
            if demand_key not in f:
                logger.warning(f"Missing {demand_key} in HDF5 file")
                continue
                
            daily_demand = f[demand_key][0]  # Shape is (1, num_days)
            
            # Create M5-style item ID and hierarchy
            item_id = f"SYNTH_{i:03d}_X_X_validation"  # Simplified hierarchy
            cat_id = f"SYNTH_CAT_{i % 3}"  # Rotate through 3 categories
            dept_id = f"SYNTH_DEPT_{i % 5}"  # Rotate through 5 departments
            store_id = f"SYNTH_STORE_{i % 2}"  # Rotate through 2 stores
            state_id = f"SYNTH_STATE_{i % 2}"  # Rotate through 2 states
            
            # Create row for this item
            row = {
                'id': item_id,
                'item_id': f"SYNTH_{i:03d}",
                'dept_id': dept_id,
                'cat_id': cat_id,
                'store_id': store_id,
                'state_id': state_id,
            }
            
            # Add daily sales columns (d_1, d_2, ..., d_N)
            for day_idx in range(min(num_days, len(daily_demand))):
                row[f'd_{day_idx + 1}'] = int(daily_demand[day_idx])
            
            sales_data.append(row)
    
    sales_df = pd.DataFrame(sales_data)
    logger.info(f"Created sales data with {len(sales_df)} items and {num_days} days")
    return sales_df

def create_simplified_prices(sales_df: pd.DataFrame, num_days: int, 
                           base_price: float = 10.0) -> pd.DataFrame:
    """
    Create simplified sell_prices.csv with constant placeholder prices.
    
    This allows the pipeline to run without price complexity.
    """
    logger.info("Creating simplified constant prices")
    
    price_data = []
    
    # Get unique stores and items
    stores = sales_df['store_id'].unique()
    items = sales_df['item_id'].unique()
    
    for store in stores:
        for item in items:
            for week in range(1, (num_days // 7) + 2):  # Cover all weeks
                price_data.append({
                    'store_id': store,
                    'item_id': item,
                    'wm_yr_wk': f"2011_{week:02d}",  # Placeholder week format
                    'sell_price': base_price  # Constant price
                })
    
    prices_df = pd.DataFrame(price_data)
    logger.info(f"Created prices data with {len(prices_df)} rows")
    return prices_df

def convert_hdf5_to_m5(hdf5_path: str, output_dir: str, 
                       start_date: str = "2011-01-29",
                       include_prices: bool = False):
    """
    Main conversion function from HDF5 to simplified M5 format.
    """
    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting {hdf5_path} to M5 format in {output_dir}")
    
    # Read HDF5 to get dimensions
    with h5py.File(hdf5_path, 'r') as f:
        # Find number of items and days from first item
        first_item_key = None
        for key in f.keys():
            if key.startswith('daily_demand_item_'):
                first_item_key = key
                break
        
        if first_item_key is None:
            raise ValueError("No daily_demand_item_* datasets found in HDF5 file")
        
        # Get dimensions
        first_item_data = f[first_item_key]
        num_days = first_item_data.shape[1]
        
        # Count items
        num_items = sum(1 for key in f.keys() if key.startswith('daily_demand_item_'))
    
    logger.info(f"Found {num_items} items with {num_days} days each")
    
    # Create calendar
    calendar_df = create_simplified_calendar(start_date, num_days)
    calendar_path = output_dir / "calendar.csv"
    calendar_df.to_csv(calendar_path, index=False)
    logger.info(f"Saved calendar to {calendar_path}")
    
    # Create sales data
    sales_df = create_sales_data_from_hdf5(str(hdf5_path), num_items, num_days)
    sales_path = output_dir / "sales_train_validation.csv"
    sales_df.to_csv(sales_path, index=False)
    logger.info(f"Saved sales data to {sales_path}")
    
    # Create prices if requested
    if include_prices:
        prices_df = create_simplified_prices(sales_df, num_days)
        prices_path = output_dir / "sell_prices.csv"
        prices_df.to_csv(prices_path, index=False)
        logger.info(f"Saved prices to {prices_path}")
    else:
        logger.info("Skipping sell_prices.csv creation")
    
    logger.info("Conversion completed successfully")
    
    # Print summary
    print(f"\nConversion Summary:")
    print(f"  Input: {hdf5_path}")
    print(f"  Output: {output_dir}")
    print(f"  Items: {num_items}")
    print(f"  Days: {num_days}")
    print(f"  Files created:")
    print(f"    - calendar.csv ({len(calendar_df)} rows)")
    print(f"    - sales_train_validation.csv ({len(sales_df)} rows)")
    if include_prices:
        print(f"    - sell_prices.csv ({len(prices_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 synthetic data to simplified M5 format")
    parser.add_argument("hdf5_path", help="Path to input HDF5 file from gen_mc_data.py")
    parser.add_argument("output_dir", help="Directory to save M5 CSV files")
    parser.add_argument("--start-date", default="2011-01-29", help="Start date for calendar (YYYY-MM-DD)")
    parser.add_argument("--include-prices", action="store_true", help="Generate sell_prices.csv with constant prices")
    
    args = parser.parse_args()
    
    convert_hdf5_to_m5(args.hdf5_path, args.output_dir, args.start_date, args.include_prices)