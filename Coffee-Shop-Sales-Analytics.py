"""
COFFEE SHOP SALES ANALYTICS - COMPLETE ANALYSIS REPORT
Author: Nikita Jakovlev
Repository: https://github.com/DaddyIPanda/Sales---Sissejuhatus-andmeteadusess

Purpose: Analyze real coffee shop sales data to extract actionable business insights
Features:
1. Automatically downloads data from Kaggle datasets
2. Cleans and prepares data for analysis
3. Analyzes sales patterns, customer behavior, and product performance
4. Generates visualizations and data-driven recommendations
5. Handles various data formats and adapts to available columns
"""

# Import required libraries - each has a specific purpose
import kagglehub  # For downloading datasets from Kaggle
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating charts and graphs
import seaborn as sns  # For enhanced visualizations
import numpy as np  # For numerical operations
import os  # For file system operations
import json  # For handling JSON data
import warnings  # For managing warning messages
from sklearn.ensemble import RandomForestRegressor  # For machine learning predictions
from sklearn.metrics import mean_absolute_percentage_error  # For model evaluation

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

print("=" * 70)
print("COFFEE SHOP SALES ANALYTICS - DATA-DRIVEN ANALYSIS")
print("=" * 70)

# ============================================================================
# SECTION 1: DATA DOWNLOAD AND LOADING
# ============================================================================
print("\n1. DOWNLOADING AND LOADING DATA FROM KAGGLE")
print("-" * 50)

print("Downloading datasets from Kaggle...")
try:
    # Download two coffee shop datasets from Kaggle
    path1 = kagglehub.dataset_download("ahmedabbas757/coffee-sales")
    path2 = kagglehub.dataset_download("keremkarayaz/coffee-shop-sales")
    print("Success: Dataset 1 downloaded to:", path1)
    print("Success: Dataset 2 downloaded to:", path2)
    
    # Store both dataset paths for processing
    dataset_paths = [path1, path2]
    
except Exception as e:
    # Fallback to local data if Kaggle download fails
    print("Warning: Error downloading datasets:", e)
    print("Falling back to local data...")
    
    # Search for local data files in current directory
    local_files = []
    for file in os.listdir('.'):
        # Check for common data file extensions
        if file.lower().endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather')):
            local_files.append(os.path.abspath(file))
    
    if not local_files:
        print("Error: No local data files found. Exiting.")
        exit()
    
    dataset_paths = ['.']  # Use current directory as data source

# ============================================================================
# SECTION 2: FINDING DATA FILES
# ============================================================================
print("\n2. SEARCHING FOR DATA FILES")
print("-" * 50)

def find_data_files(path):
    """
    Search for data files in a directory and its subdirectories.
    This function recursively walks through folders to find files with
    common data formats like CSV, Excel, JSON, etc.
    """
    data_files = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_lower = file.lower()
                # Check if file has a data format extension
                if any(file_lower.endswith(ext) for ext in [
                    '.csv', '.xlsx', '.xls', '.json', '.parquet', 
                    '.feather', '.pkl', '.h5', '.hdf5', '.txt', '.dat'
                ]):
                    full_path = os.path.join(root, file)
                    data_files.append(full_path)
    return data_files

# Search for data files in downloaded datasets
all_data_files = []

for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):
        files = find_data_files(dataset_path)
        print(f"\nFound {len(files)} data files in {os.path.basename(dataset_path)}:")
        
        # Count files by type for summary
        file_types = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        for ext, count in file_types.items():
            print(f"  - {ext}: {count} files")
        
        all_data_files.extend(files)

# Also check current directory if no files found in datasets
if not all_data_files:
    local_files = find_data_files('.')
    if local_files:
        print(f"\nFound {len(local_files)} data files in current directory:")
        for file in local_files[:10]:  # Show first 10 files
            print(f"  - {os.path.basename(file)}")
        all_data_files.extend(local_files)

if not all_data_files:
    print("Error: No data files found. Exiting.")
    exit()

print(f"\nTotal data files found: {len(all_data_files)}")

# ============================================================================
# SECTION 3: LOADING DATA FILES
# ============================================================================
print("\n3. ATTEMPTING TO LOAD DATA FILES")
print("-" * 50)

def load_data_file(file_path):
    """
    Try to load a data file with various formats.
    This function attempts to read files in different formats and
    returns the data along with the file type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
            return df, 'CSV'
        
        elif ext in ['.xlsx', '.xls']:
            # Excel files may have multiple sheets - we try to find the best one
            try:
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) == 1:
                    df = pd.read_excel(file_path)
                else:
                    # Try to find the most promising sheet with transaction data
                    sheet_info = {}
                    for sheet in excel_file.sheet_names:
                        try:
                            sheet_df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                            # Check if it looks like transaction data (has enough columns and rows)
                            if len(sheet_df.columns) >= 3 and sheet_df.shape[0] > 0:
                                sheet_info[sheet] = sheet_df.shape
                        except:
                            continue
                    
                    if sheet_info:
                        # Pick the sheet with the most columns (likely to be richest data)
                        best_sheet = max(sheet_info.keys(), key=lambda x: sheet_info[x][1])
                        df = pd.read_excel(file_path, sheet_name=best_sheet)
                    else:
                        df = pd.read_excel(file_path, sheet_name=0)
                return df, 'Excel'
            except Exception as e:
                print(f"  Excel read error: {e}")
                return None, None
        
        elif ext == '.json':
            try:
                df = pd.read_json(file_path)
                return df, 'JSON'
            except:
                # Try reading as JSON lines format
                try:
                    df = pd.read_json(file_path, lines=True)
                    return df, 'JSON Lines'
                except:
                    return None, None
        
        elif ext == '.parquet':
            try:
                df = pd.read_parquet(file_path)
                return df, 'Parquet'
            except:
                return None, None
        
        elif ext == '.feather':
            try:
                df = pd.read_feather(file_path)
                return df, 'Feather'
            except:
                return None, None
        
        elif ext in ['.pkl', '.pickle']:
            try:
                df = pd.read_pickle(file_path)
                return df, 'Pickle'
            except:
                return None, None
        
        elif ext in ['.h5', '.hdf5']:
            try:
                df = pd.read_hdf(file_path)
                return df, 'HDF5'
            except:
                return None, None
        
        elif ext in ['.txt', '.dat']:
            # Try different delimiters for text files
            try:
                df = pd.read_csv(file_path, sep=',')
                return df, 'Text (CSV)'
            except:
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    return df, 'Text (TSV)'
                except:
                    try:
                        df = pd.read_csv(file_path, sep='\s+')
                        return df, 'Text (space)'
                    except:
                        return None, None
        
        else:
            return None, None
            
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None, None

# Try to load data files in order - prioritize files with relevant names
df = None
loaded_file = None
loaded_type = None

print("\nTrying to load data files...")

# First, prioritize files with coffee/sales in the name
priority_files = []
other_files = []

for file in all_data_files:
    filename = os.path.basename(file).lower()
    if any(keyword in filename for keyword in ['coffee', 'sales', 'shop', 'transaction', 'retail', 'store']):
        priority_files.append(file)
    else:
        other_files.append(file)

print(f"Found {len(priority_files)} priority files, {len(other_files)} other files")

# Try priority files first (those likely to contain coffee shop data)
for file_list, list_name in [(priority_files, "priority"), (other_files, "other")]:
    if df is not None:
        break
    
    print(f"\nTrying {list_name} files...")
    
    for i, file_path in enumerate(file_list, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(file_list)}] Trying: {filename}")
        
        loaded_df, file_type = load_data_file(file_path)
        
        if loaded_df is not None and not loaded_df.empty:
            print(f"    Success: Successfully loaded as {file_type}")
            print(f"      Shape: {loaded_df.shape}")
            print(f"      Columns: {list(loaded_df.columns)}")
            
            # Check if this looks like sales transaction data
            # Look for columns with sales-related keywords
            sales_keywords = ['transaction', 'sale', 'order', 'price', 'amount', 'total', 'revenue', 
                            'quantity', 'qty', 'product', 'item', 'store', 'location', 'date', 'time']
            
            column_matches = 0
            for col in loaded_df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in sales_keywords):
                    column_matches += 1
            
            # Check data volume
            row_count = len(loaded_df)
            col_count = len(loaded_df.columns)
            
            print(f"      Column matches with sales keywords: {column_matches}/{col_count}")
            print(f"      Data points: {row_count:,} rows x {col_count} columns")
            
            # Decision criteria: if it has sales keywords OR has significant data
            if column_matches >= 2 or (row_count > 100 and col_count >= 3):
                df = loaded_df
                loaded_file = file_path
                loaded_type = file_type
                print(f"    Success: This looks like sales data! Using this file.")
                break
            else:
                print(f"    Note: Doesn't look like sales data, trying next...")
        else:
            print(f"    Error: Could not load file")

# Last resort: load any available file if none matched criteria
if df is None and all_data_files:
    print("\nNote: No file matched sales criteria. Loading first available file...")
    for file_path in all_data_files:
        loaded_df, file_type = load_data_file(file_path)
        if loaded_df is not None and not loaded_df.empty:
            df = loaded_df
            loaded_file = file_path
            loaded_type = file_type
            print(f"Success: Loaded first available file: {os.path.basename(loaded_file)}")
            break

if df is None:
    print("Error: Could not load any data files. Exiting.")
    exit()

print(f"\nSUCCESS: Data loaded from {loaded_type} file: {os.path.basename(loaded_file)}")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nFirst 3 rows of data:")
print(df.head(3))

print("\nColumn names and data types:")
for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    non_null = df[col].count()
    null_pct = (1 - non_null/len(df)) * 100 if len(df) > 0 else 0
    print(f"  {i:2d}. {col}: {dtype} | Non-null: {non_null:,} ({null_pct:.1f}% null)")

# ============================================================================
# SECTION 4: DATA CLEANING AND PREPARATION
# ============================================================================
print("\n4. DATA CLEANING AND PREPARATION")
print("-" * 50)

# Clean column names for easier access (remove spaces, special characters)
original_columns = df.columns.tolist()
df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]
print("Cleaned column names:", list(df.columns))

# Show basic dataset information
print(f"\nBasic dataset info:")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Total missing values: {df.isnull().sum().sum():,}")

# Identify key columns by analyzing content and names
date_cols = []
time_cols = []
product_cols = []
price_cols = []
qty_cols = []
store_cols = []

print("\nAnalyzing column content to identify data types...")

for col in df.columns:
    col_lower = col.lower()
    sample_values = df[col].dropna().head(10).astype(str).tolist() if not df[col].dropna().empty else []
    
    # Check column name patterns to categorize columns
    if any(term in col_lower for term in ['date', 'datetime', 'day', 'month', 'year']):
        date_cols.append(col)
    
    elif any(term in col_lower for term in ['time', 'timestamp', 'hour', 'minute', 'second']):
        time_cols.append(col)
    
    elif any(term in col_lower for term in ['product', 'item', 'category', 'type', 'detail', 'name', 'description']):
        product_cols.append(col)
    
    elif any(term in col_lower for term in ['price', 'amount', 'cost', 'revenue', 'total', 'value', 'sum']):
        price_cols.append(col)
    
    elif any(term in col_lower for term in ['qty', 'quantity', 'count', 'number', 'units']):
        qty_cols.append(col)
    
    elif any(term in col_lower for term in ['store', 'location', 'branch', 'shop', 'outlet', 'site']):
        store_cols.append(col)
    
    # Also infer column type from data patterns if not identified by name
    if col not in date_cols + time_cols + product_cols + price_cols + qty_cols + store_cols:
        if len(sample_values) >= 3:
            # Check for date patterns in data
            date_patterns = ['/', '-', '202', '201', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            if any(pattern in ' '.join(sample_values).lower() for pattern in date_patterns):
                date_cols.append(col)
            
            # Check for price patterns in data
            elif any('$' in val or 'usd' in val.lower() or (val.replace('.', '').replace(',', '').isdigit() and '.' in val) for val in sample_values):
                price_cols.append(col)
            
            # Check for product names (text data, not too long)
            elif all(len(val) < 100 and not val.replace('.', '').replace(',', '').isdigit() for val in sample_values):
                if col not in product_cols:
                    product_cols.append(col)

print("\nIdentified columns by type:")
print(f"  Date columns: {date_cols if date_cols else 'None'}")
print(f"  Time columns: {time_cols if time_cols else 'None'}")
print(f"  Product columns: {product_cols if product_cols else 'None'}")
print(f"  Price columns: {price_cols if price_cols else 'None'}")
print(f"  Quantity columns: {qty_cols if qty_cols else 'None'}")
print(f"  Store columns: {store_cols if store_cols else 'None'}")

# Handle datetime conversion for time-based analysis
datetime_created = False
if date_cols:
    date_col = date_cols[0]
    print(f"\nProcessing date from column: {date_col}")
    
    try:
        # Convert date column to proper datetime format
        df['transaction_datetime'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        
        # Check if conversion was successful
        valid_dates = df['transaction_datetime'].notna()
        if valid_dates.any():
            datetime_created = True
            
            # Extract date components for analysis
            df.loc[valid_dates, 'date'] = df.loc[valid_dates, 'transaction_datetime'].dt.date
            df.loc[valid_dates, 'day_of_week'] = df.loc[valid_dates, 'transaction_datetime'].dt.day_name()
            df.loc[valid_dates, 'hour'] = df.loc[valid_dates, 'transaction_datetime'].dt.hour
            df.loc[valid_dates, 'month'] = df.loc[valid_dates, 'transaction_datetime'].dt.month_name()
            df.loc[valid_dates, 'is_weekend'] = df.loc[valid_dates, 'transaction_datetime'].dt.dayofweek >= 5
            
            print(f"  Success: Date conversion successful")
            print(f"    Valid dates: {valid_dates.sum():,} of {len(df):,}")
            print(f"    Date range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}")
        else:
            print(f"  Warning: No valid dates found in column")
            df['transaction_datetime'] = None
            
    except Exception as e:
        print(f"  Error: Error processing dates: {e}")
        df['transaction_datetime'] = None
else:
    print("\nNote: No date columns identified. Time-based analysis will be limited.")
    df['transaction_datetime'] = None

# Calculate total sales for each transaction
total_sales_calculated = False

# Method 1: If we have both price and quantity, calculate total sales
if price_cols and qty_cols:
    price_col = price_cols[0]
    qty_col = qty_cols[0]
    
    print(f"\nAttempting to calculate total sales from: {price_col} x {qty_col}")
    
    # Convert to numeric for calculation
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
    
    # Check if we have valid numeric data
    price_valid = df[price_col].notna()
    qty_valid = df[qty_col].notna()
    
    if price_valid.any() and qty_valid.any():
        df['total_sales'] = df[price_col] * df[qty_col]
        total_sales_calculated = True
        print(f"  Success: Total sales calculated successfully")
        print(f"    Valid calculations: {(df['total_sales'].notna()).sum():,} of {len(df):,}")
        print(f"    Total revenue: ${df['total_sales'].sum():,.2f}")
    else:
        print(f"  Warning: Not enough valid numeric data in price or quantity columns")

# Method 2: If we only have price column, use it as total sales
elif price_cols:
    price_col = price_cols[0]
    print(f"\nUsing {price_col} as total sales")
    
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    if df[price_col].notna().any():
        df['total_sales'] = df[price_col]
        total_sales_calculated = True
        print(f"  Success: Using price column as sales amount")
        print(f"    Total revenue: ${df['total_sales'].sum():,.2f}")
    else:
        print(f"  Warning: Price column doesn't contain valid numeric data")

# Method 3: Search for any numeric column that could be sales
if not total_sales_calculated:
    print("\nSearching for sales data in other columns...")
    
    # Look for any numeric column that could be sales
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out likely non-sales columns (IDs, small numbers)
    candidate_cols = []
    for col in numeric_cols:
        if 'id' not in col.lower() and 'num' not in col.lower() and 'index' not in col.lower():
            non_null = df[col].notna().sum()
            if non_null > len(df) * 0.1:  # At least 10% non-null
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Check if values look like transaction amounts
                if mean_val > 0.5 and mean_val < 1000 and std_val > 0:
                    candidate_cols.append((col, mean_val, non_null))
    
    if candidate_cols:
        # Sort by number of non-null values, then by mean
        candidate_cols.sort(key=lambda x: (x[2], x[1]), reverse=True)
        best_col = candidate_cols[0][0]
        
        df['total_sales'] = df[best_col]
        total_sales_calculated = True
        print(f"  Success: Using {best_col} as total sales (best guess)")
        print(f"    Mean value: ${df[best_col].mean():.2f}")
        print(f"    Total: ${df[best_col].sum():,.2f}")
    else:
        print("  Warning: Could not identify sales data")
        df['total_sales'] = 0

# ============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n5. EXPLORATORY DATA ANALYSIS")
print("-" * 50)

# Basic statistics about the dataset
print(f"\nDATASET OVERVIEW")
print(f"   Total records: {len(df):,}")

if total_sales_calculated and df['total_sales'].sum() > 0:
    total_revenue = df['total_sales'].sum()
    avg_transaction = df['total_sales'].mean()
    median_transaction = df['total_sales'].median()
    
    print(f"   Total sales: ${total_revenue:,.2f}")
    print(f"   Average transaction: ${avg_transaction:.2f}")
    print(f"   Median transaction: ${median_transaction:.2f}")
    print(f"   Min transaction: ${df['total_sales'].min():.2f}")
    print(f"   Max transaction: ${df['total_sales'].max():.2f}")
    
    # Analyze transaction value distribution
    print(f"\n   Transaction value distribution:")
    bins = [0, 5, 10, 15, 20, 30, 50, 100, float('inf')]
    labels = ['Under $5', '$5-10', '$10-15', '$15-20', '$20-30', '$30-50', '$50-100', 'Over $100']
    
    df['price_bin'] = pd.cut(df['total_sales'], bins=bins, labels=labels, right=False)
    bin_counts = df['price_bin'].value_counts().sort_index()
    
    for bin_label, count in bin_counts.items():
        if pd.notna(bin_label):
            pct = (count / len(df)) * 100
            print(f"     {bin_label}: {count:,} transactions ({pct:.1f}%)")

# Analyze sales by day of week
if 'day_of_week' in df.columns and df['day_of_week'].notna().any():
    print(f"\nDAY OF WEEK ANALYSIS")
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if total_sales_calculated:
        # Calculate metrics by day of week
        day_stats = df.groupby('day_of_week').agg({
            'total_sales': ['count', 'sum', 'mean', 'median']
        })
        day_stats.columns = ['transactions', 'total_revenue', 'avg_sale', 'median_sale']
    else:
        day_stats = df.groupby('day_of_week').size().to_frame('transactions')
    
    # Ensure all days are present (fill missing days with zeros)
    for day in day_order:
        if day not in day_stats.index:
            if total_sales_calculated:
                day_stats.loc[day] = [0, 0, 0, 0]
            else:
                day_stats.loc[day] = [0]
    
    day_stats = day_stats.reindex(day_order)
    
    # Print day-by-day statistics
    for day in day_order:
        transactions = int(day_stats.loc[day, 'transactions'])
        if transactions > 0:
            if total_sales_calculated:
                revenue = day_stats.loc[day, 'total_revenue']
                avg = day_stats.loc[day, 'avg_sale']
                pct_tx = (transactions / len(df)) * 100
                pct_rev = (revenue / df['total_sales'].sum()) * 100 if df['total_sales'].sum() > 0 else 0
                print(f"   {day[:3]}: {transactions:,} transactions ({pct_tx:.1f}%), ${revenue:,.0f} revenue ({pct_rev:.1f}%), average ${avg:.2f}")
            else:
                pct = (transactions / len(df)) * 100
                print(f"   {day[:3]}: {transactions:,} transactions ({pct:.1f}%)")
    
    # Weekend vs weekday analysis
    if 'is_weekend' in df.columns:
        weekend_mask = df['is_weekend'] == True
        weekend_pct = weekend_mask.mean() * 100
        
        if total_sales_calculated:
            weekend_revenue = df.loc[weekend_mask, 'total_sales'].sum()
            total_revenue = df['total_sales'].sum()
            weekend_rev_pct = (weekend_revenue / total_revenue * 100) if total_revenue > 0 else 0
            print(f"\n   Weekend (Saturday-Sunday): {weekend_pct:.1f}% of transactions, {weekend_rev_pct:.1f}% of revenue")
        else:
            print(f"\n   Weekend (Saturday-Sunday): {weekend_pct:.1f}% of transactions")

# Analyze sales by hour of day
if 'hour' in df.columns and df['hour'].notna().any():
    print(f"\nHOURLY ANALYSIS")
    
    # Group transactions by hour
    hourly_stats = df.groupby('hour').size()
    
    if len(hourly_stats) > 0:
        # Fill missing hours with zeros for complete 24-hour view
        all_hours = range(24)
        for h in all_hours:
            if h not in hourly_stats.index:
                hourly_stats[h] = 0
        
        hourly_stats = hourly_stats.sort_index()
        hourly_stats.index = hourly_stats.index.astype(int)

        # Print top 5 busiest hours
        print("   Top 5 busiest hours:")
        top_hours = hourly_stats.nlargest(5)
        for hour, count in top_hours.items():
            pct = (count / len(df)) * 100
            print(f"     {hour:02d}:00: {count:,} transactions ({pct:.1f}%)")

        # Identify peak hour
        peak_hour = hourly_stats.idxmax()
        peak_count = hourly_stats.max()
        peak_pct = (peak_count / len(df)) * 100
        print(f"\n   Peak hour: {peak_hour:02d}:00 ({peak_count:,} transactions, {peak_pct:.1f}%)")
        
        # Analyze different time periods
        morning = hourly_stats.loc[6:11].sum() if 6 in hourly_stats.index else 0
        afternoon = hourly_stats.loc[12:17].sum() if 12 in hourly_stats.index else 0
        evening = hourly_stats.loc[18:23].sum() if 18 in hourly_stats.index else 0
        night = hourly_stats.loc[0:5].sum() if 0 in hourly_stats.index else 0
        
        print(f"\n   Time period breakdown:")
        for period, count, label in [
            (morning, morning, "Morning (6am-12pm)"),
            (afternoon, afternoon, "Afternoon (12pm-6pm)"),
            (evening, evening, "Evening (6pm-12am)"),
            (night, night, "Night (12am-6am)")
        ]:
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"     {label}: {count:,} transactions ({pct:.1f}%)")

# Analyze store performance
if store_cols:
    store_col = store_cols[0]
    print(f"\nSTORE ANALYSIS")
    print(f"   Column used: {store_col}")
    
    store_values = df[store_col].value_counts()
    print(f"   Unique stores: {len(store_values)}")
    
    if total_sales_calculated and df['total_sales'].sum() > 0:
        # Calculate store-level metrics
        store_sales = df.groupby(store_col).agg({
            'total_sales': ['sum', 'count', 'mean', 'median']
        })
        store_sales.columns = ['total_revenue', 'transactions', 'avg_sale', 'median_sale']
        store_sales = store_sales.sort_values('total_revenue', ascending=False)
        
        print("\n   Top 3 stores by revenue:")
        for i, (store, row) in enumerate(store_sales.head(3).iterrows(), 1):
            revenue_pct = (row['total_revenue'] / df['total_sales'].sum()) * 100
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"     {i}. {store}")
            print(f"        Revenue: ${row['total_revenue']:,.0f} ({revenue_pct:.1f}%)")
            print(f"        Transactions: {row['transactions']:,} ({tx_pct:.1f}%)")
            print(f"        Average sale: ${row['avg_sale']:.2f}")
    else:
        store_counts = df[store_col].value_counts()
        print("\n   Top 3 stores by transactions:")
        for i, (store, count) in enumerate(store_counts.head(3).items(), 1):
            pct = (count / len(df)) * 100
            print(f"     {i}. {store}: {count:,} transactions ({pct:.1f}%)")

# Analyze product performance
if product_cols:
    product_col = product_cols[0]
    print(f"\nPRODUCT ANALYSIS")
    print(f"   Column used: {product_col}")
    
    product_counts = df[product_col].value_counts()
    print(f"   Unique products: {len(product_counts)}")
    
    print("\n   Top 5 products by frequency:")
    for i, (product, count) in enumerate(product_counts.head(5).items(), 1):
        pct = (count / len(df)) * 100
        print(f"     {i}. {product}: {count:,} transactions ({pct:.1f}%)")
    
    if total_sales_calculated:
        # Calculate revenue by product
        product_sales = df.groupby(product_col).agg({
            'total_sales': ['sum', 'count', 'mean']
        })
        product_sales.columns = ['total_revenue', 'transactions', 'avg_price']
        product_sales = product_sales.sort_values('total_revenue', ascending=False)
        
        print("\n   Top 5 products by revenue:")
        for i, (product, row) in enumerate(product_sales.head(5).iterrows(), 1):
            revenue_pct = (row['total_revenue'] / df['total_sales'].sum()) * 100
            print(f"     {i}. {product}")
            print(f"        Revenue: ${row['total_revenue']:,.0f} ({revenue_pct:.1f}%)")
            print(f"        Transactions: {row['transactions']:,}")
            print(f"        Average price: ${row['avg_price']:.2f}")
    
    # Analyze product diversity
    print(f"\n   Product diversity:")
    top_10_pct = (product_counts.head(10).sum() / len(df)) * 100
    print(f"     Top 10 products account for {top_10_pct:.1f}% of transactions")

# ============================================================================
# SECTION 6: PRODUCT TYPE ANALYSIS
# ============================================================================
print("\n6. PRODUCT TYPE ANALYSIS")
print("-" * 50)

# Identify product category, type, and detail columns
category_cols = []
type_cols = []
detail_cols = []

# Classify columns based on naming patterns
for col in df.columns:
    col_lower = col.lower()
    if 'category' in col_lower:
        category_cols.append(col)
    elif 'type' in col_lower and 'product' in col_lower:
        type_cols.append(col)
    elif 'detail' in col_lower or 'description' in col_lower:
        detail_cols.append(col)

print("Identified product columns:")
print(f"  Category columns: {category_cols if category_cols else 'None'}")
print(f"  Type columns: {type_cols if type_cols else 'None'}")
print(f"  Detail columns: {detail_cols if detail_cols else 'None'}")

# Analyze product categories if available
if category_cols:
    category_col = category_cols[0]
    print(f"\nPRODUCT CATEGORY ANALYSIS ({category_col})")
    print("-" * 40)
    
    category_stats = df.groupby(category_col).agg({
        'total_sales': ['sum', 'count', 'mean'] if total_sales_calculated else pd.Series.count
    })
    
    if total_sales_calculated:
        category_stats.columns = ['total_revenue', 'transactions', 'avg_price']
        category_stats = category_stats.sort_values('total_revenue', ascending=False)
        
        print("Product Categories by Revenue:")
        for i, (category, row) in enumerate(category_stats.iterrows(), 1):
            revenue_pct = (row['total_revenue'] / df['total_sales'].sum()) * 100
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {category}")
            print(f"     Revenue: ${row['total_revenue']:,.0f} ({revenue_pct:.1f}%)")
            print(f"     Transactions: {row['transactions']:,} ({tx_pct:.1f}%)")
            print(f"     Average Price: ${row['avg_price']:.2f}")
    else:
        category_stats.columns = ['transactions']
        category_stats = category_stats.sort_values('transactions', ascending=False)
        
        print("Product Categories by Transactions:")
        for i, (category, row) in enumerate(category_stats.iterrows(), 1):
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {category}: {row['transactions']:,} transactions ({tx_pct:.1f}%)")

# Analyze product types if available
if type_cols:
    type_col = type_cols[0]
    print(f"\nPRODUCT TYPE ANALYSIS ({type_col})")
    print("-" * 40)
    
    type_stats = df.groupby(type_col).agg({
        'total_sales': ['sum', 'count', 'mean'] if total_sales_calculated else pd.Series.count
    })
    
    if total_sales_calculated:
        type_stats.columns = ['total_revenue', 'transactions', 'avg_price']
        type_stats = type_stats.sort_values('total_revenue', ascending=False)
        
        print("Top Product Types by Revenue:")
        for i, (product_type, row) in enumerate(type_stats.head(10).iterrows(), 1):
            revenue_pct = (row['total_revenue'] / df['total_sales'].sum()) * 100
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {product_type}")
            print(f"     Revenue: ${row['total_revenue']:,.0f} ({revenue_pct:.1f}%)")
            print(f"     Transactions: {row['transactions']:,} ({tx_pct:.1f}%)")
            print(f"     Average Price: ${row['avg_price']:.2f}")
    else:
        type_stats.columns = ['transactions']
        type_stats = type_stats.sort_values('transactions', ascending=False)
        
        print("Top Product Types by Transactions:")
        for i, (product_type, row) in enumerate(type_stats.head(10).iterrows(), 1):
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {product_type}: {row['transactions']:,} transactions ({tx_pct:.1f}%)")

# Analyze product details if available
if detail_cols:
    detail_col = detail_cols[0]
    print(f"\nPRODUCT DETAIL ANALYSIS ({detail_col})")
    print("-" * 40)
    
    detail_stats = df.groupby(detail_col).agg({
        'total_sales': ['sum', 'count', 'mean'] if total_sales_calculated else pd.Series.count
    })
    
    if total_sales_calculated:
        detail_stats.columns = ['total_revenue', 'transactions', 'avg_price']
        detail_stats = detail_stats.sort_values('total_revenue', ascending=False)
        
        print("Top Product Details by Revenue:")
        for i, (detail, row) in enumerate(detail_stats.head(10).iterrows(), 1):
            revenue_pct = (row['total_revenue'] / df['total_sales'].sum()) * 100
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {detail}")
            print(f"     Revenue: ${row['total_revenue']:,.0f} ({revenue_pct:.1f}%)")
            print(f"     Transactions: {row['transactions']:,} ({tx_pct:.1f}%)")
            print(f"     Average Price: ${row['avg_price']:.2f}")
    else:
        detail_stats.columns = ['transactions']
        detail_stats = detail_stats.sort_values('transactions', ascending=False)
        
        print("Top Product Details by Transactions:")
        for i, (detail, row) in enumerate(detail_stats.head(10).iterrows(), 1):
            tx_pct = (row['transactions'] / len(df)) * 100
            print(f"  {i}. {detail}: {row['transactions']:,} transactions ({tx_pct:.1f}%)")

# ============================================================================
# SECTION 7: PRODUCT TYPE VISUALIZATIONS
# ============================================================================
print("\n7. PRODUCT TYPE VISUALIZATIONS")
print("-" * 50)

# Visualization 1: Product Category Analysis
print("\nCreating Visualization 1: Product Category Analysis...")
if category_cols:
    category_col = category_cols[0]
    category_data = df[category_col].value_counts().head(10)
    
    if len(category_data) > 0:
        categories = category_data.index.tolist()
        counts = category_data.values.tolist()
        
        # Truncate long category names for better display
        short_names = []
        for name in categories:
            name_str = str(name)
            if len(name_str) > 25:
                short_names.append(name_str[:23] + '...')
            else:
                short_names.append(name_str)
        
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left subplot: Bar chart showing top categories
        colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
        y_pos = range(len(categories))
        bars = ax7a.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        ax7a.set_title(f'Top Product Categories by Frequency', fontweight='bold', fontsize=14)
        ax7a.set_xlabel('Number of Transactions', fontsize=12)
        ax7a.set_ylabel('Product Category', fontsize=12)
        ax7a.set_yticks(y_pos)
        ax7a.set_yticklabels(short_names)
        ax7a.invert_yaxis()
        ax7a.grid(True, alpha=0.3, axis='x')
        ax7a.set_axisbelow(True)
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (counts[i] / len(df)) * 100
            ax7a.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
        
        # Right subplot: Pie chart showing distribution
        top_n = min(8, len(categories))
        pie_counts = counts[:top_n]
        pie_names = short_names[:top_n]
        other_count = sum(counts[top_n:]) if len(counts) > top_n else 0
        
        if other_count > 0:
            pie_counts.append(other_count)
            pie_names.append('Other Categories')
        
        colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(pie_counts)))
        wedges, texts, autotexts = ax7b.pie(pie_counts, labels=pie_names, colors=colors_pie,
                                           autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                                           startangle=90, wedgeprops=dict(width=0.5))
        
        ax7b.set_title(f'Product Category Distribution', fontweight='bold', fontsize=14)
        
        # Add center text with total transactions
        total_transactions = sum(counts)
        center_text = f"Total:\n{total_transactions:,}\ntransactions"
        ax7b.text(0, 0, center_text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Adjust text properties for readability
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('product_categories.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Success: Saved product_categories.png")
    else:
        print("Note: No category data available")
else:
    print("Note: Category data not available")

# Visualization 2: Product Type Analysis
print("\nCreating Visualization 2: Product Type Analysis...")
if type_cols:
    type_col = type_cols[0]
    type_data = df[type_col].value_counts().head(10)
    
    if len(type_data) > 0:
        types = type_data.index.tolist()
        counts = type_data.values.tolist()
        
        # Truncate long type names for better display
        short_names = []
        for name in types:
            name_str = str(name)
            if len(name_str) > 25:
                short_names.append(name_str[:23] + '...')
            else:
                short_names.append(name_str)
        
        fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left subplot: Bar chart of product types by frequency
        colors = plt.cm.Paired(np.linspace(0, 1, len(types)))
        y_pos = range(len(types))
        bars = ax8a.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        ax8a.set_title(f'Top Product Types by Frequency', fontweight='bold', fontsize=14)
        ax8a.set_xlabel('Number of Transactions', fontsize=12)
        ax8a.set_ylabel('Product Type', fontsize=12)
        ax8a.set_yticks(y_pos)
        ax8a.set_yticklabels(short_names)
        ax8a.invert_yaxis()
        ax8a.grid(True, alpha=0.3, axis='x')
        ax8a.set_axisbelow(True)
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (counts[i] / len(df)) * 100
            ax8a.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
        
        # Right subplot: Revenue by product type (if revenue data available)
        if total_sales_calculated and df['total_sales'].sum() > 0:
            type_revenue = df.groupby(type_col)['total_sales'].sum().sort_values(ascending=False).head(10)
            type_rev_values = type_revenue.values
            type_rev_labels = [str(label)[:20] + '...' if len(str(label)) > 20 else str(label) for label in type_revenue.index]
            
            colors_rev = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(type_rev_values)))
            y_pos_rev = range(len(type_rev_values))
            bars_rev = ax8b.barh(y_pos_rev, type_rev_values, color=colors_rev, edgecolor='black', linewidth=0.5)
            
            ax8b.set_title(f'Top Product Types by Revenue', fontweight='bold', fontsize=14)
            ax8b.set_xlabel('Total Revenue ($)', fontsize=12)
            ax8b.set_ylabel('Product Type', fontsize=12)
            ax8b.set_yticks(y_pos_rev)
            ax8b.set_yticklabels(type_rev_labels)
            ax8b.invert_yaxis()
            ax8b.grid(True, alpha=0.3, axis='x')
            ax8b.set_axisbelow(True)
            
            # Format x-axis as currency
            ax8b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add value labels to revenue bars
            for i, bar in enumerate(bars_rev):
                width = bar.get_width()
                pct = (type_rev_values[i] / df['total_sales'].sum()) * 100
                ax8b.text(width + max(type_rev_values)*0.01, bar.get_y() + bar.get_height()/2,
                         f'${width:,.0f} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('product_types.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Success: Saved product_types.png")
    else:
        print("Note: No type data available")
else:
    print("Note: Type data not available")

# Visualization 3: Product Detail Analysis
print("\nCreating Visualization 3: Product Detail Analysis...")
if detail_cols:
    detail_col = detail_cols[0]
    detail_data = df[detail_col].value_counts().head(15)
    
    if len(detail_data) > 0:
        details = detail_data.index.tolist()
        counts = detail_data.values.tolist()
        
        # Truncate long detail names for better display
        short_names = []
        for name in details:
            name_str = str(name)
            if len(name_str) > 30:
                short_names.append(name_str[:28] + '...')
            else:
                short_names.append(name_str)
        
        fig9, ax9 = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart for product details
        colors = plt.cm.tab20c(np.linspace(0, 1, len(details)))
        y_pos = range(len(details))
        bars = ax9.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        ax9.set_title(f'Top Product Details by Frequency', fontweight='bold', fontsize=14)
        ax9.set_xlabel('Number of Transactions', fontsize=12)
        ax9.set_ylabel('Product Detail', fontsize=12)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(short_names)
        ax9.invert_yaxis()
        ax9.grid(True, alpha=0.3, axis='x')
        ax9.set_axisbelow(True)
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (counts[i] / len(df)) * 100
            ax9.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('product_details.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Success: Saved product_details.png")
    else:
        print("Note: No detail data available")
else:
    print("Note: Detail data not available")

# Visualization 4: Product Hierarchy Analysis (if we have multiple levels)
print("\nCreating Visualization 4: Product Hierarchy Analysis...")
if category_cols and type_cols:
    # Create a cross-tabulation of category and type to see combinations
    cross_tab = pd.crosstab(df[category_cols[0]], df[type_cols[0]])
    
    if not cross_tab.empty and cross_tab.sum().sum() > 0:
        # Extract top category-type combinations
        category_types = []
        for category in cross_tab.index:
            for ptype in cross_tab.columns:
                count = cross_tab.loc[category, ptype]
                if count > 0:
                    category_types.append({
                        'category': category,
                        'type': ptype,
                        'count': count
                    })
        
        # Sort by count and take top 15
        category_types.sort(key=lambda x: x['count'], reverse=True)
        category_types = category_types[:15]
        
        # Prepare data for visualization
        categories = [item['category'] for item in category_types]
        types = [item['type'] for item in category_types]
        counts = [item['count'] for item in category_types]
        
        # Create labels for the chart (combine category and type)
        labels = []
        for item in category_types:
            label = f"{item['category'][:15]}{'...' if len(item['category']) > 15 else ''} | "
            label += f"{item['type'][:15]}{'...' if len(item['type']) > 15 else ''}"
            labels.append(label)
        
        fig10, ax10 = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(category_types)))
        y_pos = range(len(category_types))
        bars = ax10.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
        
        ax10.set_title(f'Top Category-Type Combinations', fontweight='bold', fontsize=14)
        ax10.set_xlabel('Number of Transactions', fontsize=12)
        ax10.set_ylabel('Category | Type', fontsize=12)
        ax10.set_yticks(y_pos)
        ax10.set_yticklabels(labels)
        ax10.invert_yaxis()
        ax10.grid(True, alpha=0.3, axis='x')
        ax10.set_axisbelow(True)
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (counts[i] / len(df)) * 100
            ax10.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
        
        # Add legend for color coding if not too many categories
        unique_categories = list(set(categories))
        if len(unique_categories) <= 8:
            legend_elements = []
            color_map = plt.cm.viridis(np.linspace(0.2, 0.9, len(unique_categories)))
            for i, cat in enumerate(unique_categories[:8]):
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color_map[i], 
                                                    label=cat[:20] + ('...' if len(cat) > 20 else '')))
            ax10.legend(handles=legend_elements, title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('product_hierarchy.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("Success: Saved product_hierarchy.png")
    else:
        print("Note: No cross-tabulation data available")
elif category_cols or type_cols:
    print("Note: Need both category and type columns for hierarchy analysis")
else:
    print("Note: Category and type data not available for hierarchy analysis")

# ============================================================================
# SECTION 8: DAILY SPENDING ANALYSIS & VISUALIZATION
# ============================================================================
print("\n8. DAILY SPENDING ANALYSIS")
print("-" * 50)

# Try to detect date column for time series analysis
possible_date_cols = ['timestamp', 'datetime', 'date', 'order_time', 'transaction_date', 'sale_date', 'purchase_date']
date_col = None
for c in possible_date_cols:
    if c in df.columns:
        date_col = c
        break

# Try to detect transaction value column for spending analysis
if 'total_sales' in df.columns and total_sales_calculated:
    tx_col = 'total_sales'
else:
    possible_tx_cols = ['total_sales', 'transaction_value', 'transaction_total', 'amount', 'sale_amount', 'price_total', 'revenue']
    tx_col = None
    for c in possible_tx_cols:
        if c in df.columns:
            tx_col = c
            break

if date_col and tx_col:
    try:
        # Convert to datetime for time series analysis
        df['_date_temp'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filter out invalid dates
        valid_dates = df['_date_temp'].notna()
        
        if valid_dates.any():
            # Extract date part only (without time)
            df.loc[valid_dates, '_date_only'] = df.loc[valid_dates, '_date_temp'].dt.date
            
            # Aggregate data by date to get daily totals
            daily_spending = df.groupby('_date_only')[tx_col].agg(['sum', 'count']).reset_index()
            daily_spending.columns = ['date', 'total_daily_spend', 'transaction_count']
            
            # Sort chronologically
            daily_spending = daily_spending.sort_values('date')
            
            # Create 2-panel visualization for daily spending analysis
            fig11, (ax11a, ax11b) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Panel 1: Daily Total Spending Line Chart
            ax11a.plot(daily_spending['date'], daily_spending['total_daily_spend'], 
                     marker='o', linewidth=2, markersize=4, color='blue', alpha=0.7)
            
            # Calculate and plot 7-day moving average for trend analysis
            if len(daily_spending) >= 7:
                daily_spending['7_day_avg'] = daily_spending['total_daily_spend'].rolling(window=7, min_periods=1).mean()
                ax11a.plot(daily_spending['date'], daily_spending['7_day_avg'], 
                         linewidth=3, color='red', alpha=0.8, label='7-Day Moving Average')
                ax11a.legend()
            
            ax11a.set_title('Daily Customer Spending Over Time', fontweight='bold', fontsize=16)
            ax11a.set_xlabel('Date', fontsize=12)
            ax11a.set_ylabel('Total Daily Spend ($)', fontsize=12)
            ax11a.grid(True, alpha=0.3)
            ax11a.tick_params(axis='x', rotation=45)
            
            # Format y-axis as currency
            ax11a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Highlight top 3 spending days with annotations
            if len(daily_spending) > 0:
                top_days = daily_spending.nlargest(3, 'total_daily_spend')
                for _, row in top_days.iterrows():
                    ax11a.annotate(f'${row["total_daily_spend"]:,.0f}', 
                                 (row['date'], row['total_daily_spend']),
                                 xytext=(0, 10), textcoords='offset points',
                                 ha='center', va='bottom', fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Panel 2: Daily Transactions and Average Spend
            if 'total_daily_spend' in daily_spending.columns and 'transaction_count' in daily_spending.columns:
                # Calculate average spend per transaction for each day
                daily_spending['avg_spend_per_tx'] = daily_spending['total_daily_spend'] / daily_spending['transaction_count']
                
                # Bar chart for transaction count
                bars = ax11b.bar(daily_spending['date'], daily_spending['transaction_count'], 
                               alpha=0.7, color='green', label='Transaction Count', width=0.8)
                
                # Second y-axis for average spend per transaction
                ax11b2 = ax11b.twinx()
                ax11b2.plot(daily_spending['date'], daily_spending['avg_spend_per_tx'], 
                          color='orange', linewidth=2, marker='s', markersize=4, label='Average Spend per Transaction')
                
                ax11b.set_title('Daily Transactions and Average Spend', fontweight='bold', fontsize=16)
                ax11b.set_xlabel('Date', fontsize=12)
                ax11b.set_ylabel('Number of Transactions', fontsize=12, color='green')
                ax11b2.set_ylabel('Average Spend per Transaction ($)', fontsize=12, color='orange')
                
                ax11b.tick_params(axis='x', rotation=45)
                ax11b.grid(True, alpha=0.3, axis='y')
                
                # Format average spend axis as currency
                ax11b2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
                
                # Combine legends from both axes
                lines1, labels1 = ax11b.get_legend_handles_labels()
                lines2, labels2 = ax11b2.get_legend_handles_labels()
                ax11b.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            plt.savefig('daily_spending_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            print("Success: Saved daily_spending_analysis.png")
            
            # Print summary statistics
            print(f"\nDAILY SPENDING SUMMARY:")
            print(f"   Total days analyzed: {len(daily_spending):,}")
            print(f"   Average daily spend: ${daily_spending['total_daily_spend'].mean():,.2f}")
            print(f"   Median daily spend: ${daily_spending['total_daily_spend'].median():,.2f}")
            print(f"   Highest daily spend: ${daily_spending['total_daily_spend'].max():,.2f}")
            print(f"   Lowest daily spend: ${daily_spending['total_daily_spend'].min():,.2f}")
            print(f"   Total revenue period: ${daily_spending['total_daily_spend'].sum():,.2f}")
            
            # Save daily spending data to CSV for further analysis
            daily_spending.to_csv('daily_spending_data.csv', index=False)
            print("Success: Saved daily_spending_data.csv")
            
        else:
            print("Note: Date column found but no valid dates in data")
            
    except Exception as e:
        print(f"Error: Error creating daily spending graph: {e}")
        print(f"   Error details: {str(e)}")
else:
    if not date_col:
        print("Note: No date column found for daily analysis")
    if not tx_col:
        print("Note: No transaction value column found for daily analysis")

# Clean up temporary columns created for analysis
if '_date_temp' in df.columns:
    df.drop(columns=['_date_temp'], inplace=True, errors='ignore')
if '_date_only' in df.columns:
    df.drop(columns=['_date_only'], inplace=True, errors='ignore')

# ============================================================================
# SECTION 9: PRODUCT INSIGHTS GENERATION
# ============================================================================
print("\n9. PRODUCT INSIGHTS GENERATION")
print("-" * 50)

product_insights = []

# Generate insights about product categories
if category_cols:
    category_col = category_cols[0]
    category_counts = df[category_col].value_counts()
    if not category_counts.empty:
        top_category = category_counts.idxmax()
        top_category_count = category_counts.max()
        top_category_pct = (top_category_count / len(df)) * 100
        
        if len(category_counts) == 1:
            product_insights.append(f"Only one product category: {top_category} ({top_category_pct:.1f}% of transactions)")
        else:
            product_insights.append(f"Top product category: {top_category} ({top_category_pct:.1f}% of transactions)")
            
            # Assess category diversity
            if len(category_counts) >= 3:
                product_insights.append(f"Product diversity: {len(category_counts)} different categories")
            
            # Check if one category dominates the business
            if top_category_pct > 50:
                product_insights.append(f"{top_category} dominates with {top_category_pct:.1f}% of transactions")

# Generate insights about product types
if type_cols:
    type_col = type_cols[0]
    type_counts = df[type_col].value_counts()
    if not type_counts.empty:
        top_type = type_counts.idxmax()
        top_type_count = type_counts.max()
        top_type_pct = (top_type_count / len(df)) * 100
        
        product_insights.append(f"Most popular product type: {top_type} ({top_type_pct:.1f}% of transactions)")
        
        # Check product type concentration
        top_3_types_pct = type_counts.head(3).sum() / len(df) * 100
        if top_3_types_pct > 70:
            product_insights.append(f"Top 3 product types account for {top_3_types_pct:.1f}% of transactions (high concentration)")

# Generate insights about product details/variety
if detail_cols:
    detail_col = detail_cols[0]
    detail_counts = df[detail_col].value_counts()
    if not detail_counts.empty:
        top_detail = detail_counts.idxmax()
        top_detail_count = detail_counts.max()
        top_detail_pct = (top_detail_count / len(df)) * 100
        
        # Assess product variety based on number of unique details
        if len(detail_counts) > 50:
            product_insights.append(f"Wide product variety: {len(detail_counts)} different product details")
        elif len(detail_counts) > 20:
            product_insights.append(f"Moderate product variety: {len(detail_counts)} different product details")
        else:
            product_insights.append(f"Limited product variety: {len(detail_counts)} different product details")

# Print all generated product insights
if product_insights:
    print("\nPRODUCT INSIGHTS:")
    print("-" * 40)
    for i, insight in enumerate(product_insights, 1):
        print(f"{i}. {insight}")
else:
    print("\nNote: No specific product insights could be generated")

# ============================================================================
# SECTION 10: PRODUCT-BASED RECOMMENDATIONS
# ============================================================================
print("\n10. PRODUCT-BASED RECOMMENDATIONS")
print("-" * 50)

product_recommendations = []

# Inventory management recommendations
if category_cols:
    category_col = category_cols[0]
    top_categories = df[category_col].value_counts().head(3).index.tolist()
    if top_categories:
        category_names = [str(c)[:20] + '...' if len(str(c)) > 20 else str(c) for c in top_categories]
        product_recommendations.append(f"Focus inventory management on top categories: {', '.join(category_names)}")

# Product development recommendations
if type_cols:
    type_col = type_cols[0]
    type_counts = df[type_col].value_counts()
    if not type_counts.empty:
        # Identify underrepresented product types
        avg_type_count = len(df) / len(type_counts) if len(type_counts) > 0 else 0
        underrepresented = []
        
        for ptype, count in type_counts.items():
            if count < avg_type_count * 0.3 and count > 0:  # Less than 30% of average
                underrepresented.append(ptype)
        
        if underrepresented and len(underrepresented) <= 3:
            under_names = [str(u)[:15] + '...' if len(str(u)) > 15 else str(u) for u in underrepresented]
            product_recommendations.append(f"Consider promoting underperforming product types: {', '.join(under_names)}")

# Pricing strategy recommendations
if detail_cols and total_sales_calculated:
    detail_col = detail_cols[0]
    # Find products with highest average price (premium products)
    product_prices = df.groupby(detail_col)['total_sales'].mean().sort_values(ascending=False)
    if len(product_prices) > 0:
        premium_products = product_prices.head(3).index.tolist()
        if premium_products:
            premium_names = [str(p)[:15] + '...' if len(str(p)) > 15 else str(p) for p in premium_products]
            avg_premium = product_prices.head(3).mean()
            product_recommendations.append(f"Highlight premium products: {', '.join(premium_names)} (average price: ${avg_premium:.2f})")

# Product bundling recommendations
if category_cols and type_cols:
    category_col = category_cols[0]
    type_col = type_cols[0]
    
    # Find common category-type combinations for potential bundles
    cross_tab = pd.crosstab(df[category_col], df[type_col])
    if not cross_tab.empty:
        bundle_suggestions = []
        for category in cross_tab.index:
            top_type_for_category = cross_tab.loc[category].idxmax()
            count = cross_tab.loc[category, top_type_for_category]
            if count > len(df) * 0.01:  # At least 1% of transactions
                bundle_suggestions.append(f"{category} with {top_type_for_category}")
        
        if bundle_suggestions and len(bundle_suggestions) <= 3:
            product_recommendations.append(f"Consider product bundles: {'; '.join(bundle_suggestions[:3])}")

# Print all product recommendations
if product_recommendations:
    print("\nPRODUCT RECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(product_recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("\nNote: No specific product recommendations without sufficient data")

# ============================================================================
# SECTION 11: VISUALIZATION SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION SUMMARY")
print("=" * 70)
print("Created the following visualizations:")
print(" 1. Product Categories (product_categories.png)")
print(" 2. Product Types (product_types.png)")
print(" 3. Product Details (product_details.png)")
print(" 4. Product Hierarchy (product_hierarchy.png)")
print(" 5. Daily Spending Analysis (daily_spending_analysis.png)")

# ============================================================================
# SECTION 12: KEY INSIGHTS AND BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n12. KEY BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("-" * 50)

print("\nKEY INSIGHTS BASED ON DATA ANALYSIS:")
print("-" * 40)

insights = []

# Time-based business insights
if datetime_created:
    if 'day_of_week' in df.columns:
        day_counts = df['day_of_week'].value_counts()
        if not day_counts.empty:
            busiest_day = day_counts.idxmax()
            busiest_day_count = day_counts.max()
            busiest_day_pct = (busiest_day_count / len(df)) * 100
            insights.append(f"Busiest day is {busiest_day} with {busiest_day_pct:.1f}% of transactions")
    
    if 'hour' in df.columns and df['hour'].notna().any():
        hour_counts = df['hour'].value_counts()
        if not hour_counts.empty:
            peak_hour = hour_counts.idxmax()
            peak_hour_count = hour_counts.max()
            peak_hour_pct = (peak_hour_count / len(df)) * 100
            insights.append(f"Peak hour is {peak_hour}:00 with {peak_hour_pct:.1f}% of transactions")

# Store performance insights
if store_cols:
    store_col = store_cols[0]
    if total_sales_calculated and df['total_sales'].sum() > 0:
        store_revenue = df.groupby(store_col)['total_sales'].sum()
        if not store_revenue.empty:
            top_store = store_revenue.idxmax()
            top_store_revenue = store_revenue.max()
            top_store_pct = (top_store_revenue / df['total_sales'].sum()) * 100
            insights.append(f"Top-performing store '{top_store}' generates {top_store_pct:.1f}% of total revenue")
    else:
        store_counts = df[store_col].value_counts()
        if not store_counts.empty:
            busiest_store = store_counts.idxmax()
            busiest_store_count = store_counts.max()
            busiest_store_pct = (busiest_store_count / len(df)) * 100
            insights.append(f"Busiest store '{busiest_store}' handles {busiest_store_pct:.1f}% of transactions")

# Product performance insights
if product_cols:
    product_col = product_cols[0]
    product_counts = df[product_col].value_counts()
    if not product_counts.empty:
        top_product = product_counts.idxmax()
        top_product_count = product_counts.max()
        top_product_pct = (top_product_count / len(df)) * 100
        insights.append(f"Most popular product '{top_product}' accounts for {top_product_pct:.1f}% of transactions")

# Sales performance insights
if total_sales_calculated and df['total_sales'].sum() > 0:
    avg_transaction = df['total_sales'].mean()
    total_revenue = df['total_sales'].sum()
    
    insights.append(f"Average transaction value: ${avg_transaction:.2f}")
    insights.append(f"Total revenue analyzed: ${total_revenue:,.2f}")
    
    # Small transaction analysis
    small_tx = (df['total_sales'] < 5).sum()
    if small_tx > 0:
        small_tx_pct = (small_tx / len(df)) * 100
        insights.append(f"{small_tx_pct:.1f}% of transactions are under $5")

# Print all business insights
if insights:
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
else:
    print("Note: No specific insights could be generated from available data.")

print("\nDATA-DRIVEN BUSINESS RECOMMENDATIONS:")
print("-" * 40)

recommendations = []

# Staffing optimization recommendations
if 'hour' in df.columns and df['hour'].notna().any():
    hourly_counts = df.groupby('hour').size()
    if len(hourly_counts) > 0:
        # Identify hours with above-average traffic for staffing
        avg_hourly = hourly_counts.mean()
        busy_hours = hourly_counts[hourly_counts > avg_hourly].index.tolist()
        
        if busy_hours:
            # Group consecutive busy hours into periods
            busy_periods = []
            current_period = []
            for h in sorted(busy_hours):
                if not current_period or h == current_period[-1] + 1:
                    current_period.append(h)
                else:
                    if len(current_period) >= 2:
                        busy_periods.append(current_period)
                    current_period = [h]
            
            if current_period and len(current_period) >= 2:
                busy_periods.append(current_period)
            
            if busy_periods:
                rec = "Schedule additional staff during peak periods: "
                periods_str = []
                for period in busy_periods[:2]:  # Top 2 periods
                    start = period[0]
                    end = period[-1] + 1
                    periods_str.append(f"{start}:00-{end}:00")
                rec += ", ".join(periods_str)
                recommendations.append(rec)

# Inventory optimization recommendations
if product_cols:
    product_col = product_cols[0]
    top_products = df[product_col].value_counts().head(3).index.tolist()
    if top_products:
        product_names = [str(p)[:20] + '...' if len(str(p)) > 20 else str(p) for p in top_products]
        recommendations.append(f"Ensure adequate stock of top-selling products: {', '.join(product_names)}")

# Store management recommendations
if store_cols and len(df[store_cols[0]].unique()) > 1:
    store_col = store_cols[0]
    if total_sales_calculated:
        # Identify store with lowest average transaction value
        store_avg = df.groupby(store_col)['total_sales'].mean()
        if len(store_avg) > 0:
            lowest_avg_store = store_avg.idxmin()
            lowest_avg = store_avg.min()
            avg_all = df['total_sales'].mean()
            
            if lowest_avg < avg_all * 0.8:  # 20% below average
                recommendations.append(f"Review pricing and upselling at '{lowest_avg_store}' (average: ${lowest_avg:.2f} vs overall average: ${avg_all:.2f})")

# Product strategy optimization
if product_cols and total_sales_calculated:
    product_col = product_cols[0]
    # Find products with high frequency but low average price
    product_stats = df.groupby(product_col).agg({
        'total_sales': ['count', 'mean']
    })
    product_stats.columns = ['frequency', 'avg_price']
    
    if len(product_stats) > 0:
        # Normalize metrics for comparison
        product_stats['freq_norm'] = (product_stats['frequency'] - product_stats['frequency'].min()) / (product_stats['frequency'].max() - product_stats['frequency'].min())
        product_stats['price_norm'] = (product_stats['avg_price'] - product_stats['avg_price'].min()) / (product_stats['avg_price'].max() - product_stats['avg_price'].min())
        
        # Find high frequency, low price items (opportunity to increase price)
        product_stats['opportunity_score'] = product_stats['freq_norm'] * (1 - product_stats['price_norm'])
        opportunity_items = product_stats.nlargest(2, 'opportunity_score').index.tolist()
        
        if opportunity_items:
            item_names = [str(i)[:15] + '...' if len(str(i)) > 15 else str(i) for i in opportunity_items]
            recommendations.append(f"Consider price optimization for frequently purchased items: {', '.join(item_names)}")

# Print all business recommendations
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("Note: No specific recommendations without sufficient data.")

# ============================================================================
# SECTION 13: EXECUTIVE SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
print("=" * 70)

print(f"\nEXECUTIVE SUMMARY:")
print(f"Data source: {os.path.basename(loaded_file)} ({loaded_type})")
print(f"Records analyzed: {len(df):,}")
if total_sales_calculated and df['total_sales'].sum() > 0:
    print(f"Total revenue: ${df['total_sales'].sum():,.2f}")
    print(f"Average transaction: ${df['total_sales'].mean():.2f}")

print(f"\nKEY OUTPUTS:")
print(f"Multiple visualization files (.png)")
print(f"CSV data exports for further analysis")
print(f"Data-driven insights and recommendations")

print(f"\nMETHODOLOGY:")
print(f"1. Automated data download from Kaggle datasets")
print(f"2. Intelligent file format detection and loading")
print(f"3. Automatic column identification and data cleaning")
print(f"4. Comprehensive exploratory data analysis")
print(f"5. Pattern extraction and business insight generation")
print(f"6. Visualization creation and recommendation formulation")

print(f"\nLIMITATIONS AND ASSUMPTIONS:")
print(f"Analysis limited to available columns in the dataset")
print(f"Results depend on data quality and completeness")
print(f"No assumptions beyond what the data explicitly contains")
print(f"Automatic column detection may not be 100% accurate")

print(f"\nNEXT STEPS:")
print(f"Review generated visualizations in the current directory")
print(f"Examine exported CSV files for detailed data")
print(f"Implement recommendations based on data insights")
print(f"Consider additional analysis with more specific business questions")

print("\n" + "=" * 70)
print("END OF ANALYSIS REPORT")
print("=" * 70)