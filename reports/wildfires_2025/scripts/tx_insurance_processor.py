import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import glob
import os

def create_file_list(directory, pattern):
    """
    Create a list of files in the given directory that match the given pattern.
    """
    file_list = glob.glob(os.path.join(directory, pattern))
    return file_list

def build_combined_df(file_list, columns_to_select):
    """
    Build a combined DataFrame from the list of files.
    Handles duplicate column names by renaming them appropriately.
    Also handles column names with line breaks.
    Filters out rows where ZIP is not a 5-character integer.
    """
    combined_df = pd.DataFrame()
    
    # Define column name mapping to handle duplicates
    col_mapping = {
        'TOTAL PAID LOSS': 'TOTAL_PAID_LOSS',
        'FIRE': 'FIRE_LOSS',      # First occurrence of FIRE
        'TOTAL PAID CLAIMS': 'TOTAL_PAID_CLAIMS',
        'FIRE.1': 'FIRE_CLAIMS'   # Second occurrence will be labeled as FIRE.1 by pandas
    }
    
    # Create comprehensive cleanup mapping for columns with line breaks
    # and different column naming conventions across years
    cleanup_mapping = {
        # Line break variations
        'TOTAL PAID LOSS\n\n': 'TOTAL PAID LOSS',
        'TOTAL PAID CLAIMS\n': 'TOTAL PAID CLAIMS',
        
        # 2009/2012 naming variations
        'PREMIUM': 'PREMIUM IN FORCE AT END OF QTR',
        'EXPOSURE(in $1,000)': 'EXPOSURE IN FORCE AT END OF QTR ($000)',
        'TOTALPOLICY': 'POLICIES IN FORCE AT END OF QTR',
        'TOTALLOSS': 'TOTAL PAID LOSS',
        'TOTALCLAIM': 'TOTAL PAID CLAIMS',
        
        # Handle variations with line breaks
        'EXPOSURE\n(in $1,000)': 'EXPOSURE IN FORCE AT END OF QTR ($000)',
        'TOTAL\nPOLICY': 'POLICIES IN FORCE AT END OF QTR',
        'TOTAL\nLOSS': 'TOTAL PAID LOSS',
        'TOTAL\nCLAIM': 'TOTAL PAID CLAIMS'
    }
    
    # Create a reverse mapping for lookup during column validation
    reverse_mapping = {}
    for old_col, new_col in cleanup_mapping.items():
        if new_col in reverse_mapping:
            reverse_mapping[new_col].append(old_col)
        else:
            reverse_mapping[new_col] = [old_col]
    
    total_rows_processed = 0
    total_rows_kept = 0
    
    for file in file_list:
        try:
            # Extract year more robustly
            filename = os.path.basename(file)
            if filename.startswith('~$'):  # Skip temporary Excel files
                print(f"Skipping temporary file: {file}")
                continue
                
            if filename.startswith('r'):
                # Handle standard files (e.g., r2012_z.xlsx)
                year = filename[1:5]  # Extract the year part
            elif filename.startswith('New_'):
                # Handle other cases or use default extraction
                year = filename.split('_')[1][1:5]
            else:
                # Fallback case
                year = ''.join(filter(str.isdigit, filename))[:4]
                
            print(f"Processing file: {file} for year: {year}")
            # Read the file with all columns
            df = pd.read_excel(file)
            initial_row_count = len(df)
            total_rows_processed += initial_row_count
            
            # Clean up column names by removing line breaks and standardizing names
            df_columns_original = df.columns.tolist()
            # First, clean up any columns with line breaks and standardize names
            new_columns = []
            for col in df_columns_original:
                if col in cleanup_mapping:
                    new_columns.append(cleanup_mapping[col])
                elif isinstance(col, str) and '\n' in col:
                    # Remove line breaks from column names not explicitly mapped
                    clean_col = col.replace('\n', '')
                    new_columns.append(clean_col)
                else:
                    new_columns.append(col)
            
            # Rename the DataFrame columns
            df.columns = new_columns
            df_columns = df.columns.tolist()
            
            # Check for missing columns, considering both direct matches and mapped columns
            missing_columns = []
            for col in columns_to_select:
                # Check if column exists directly
                if col in df_columns:
                    continue
                
                # Check if any of the original column names map to this expected column
                if col in reverse_mapping:
                    found = False
                    for orig_col in reverse_mapping[col]:
                        if orig_col in df_columns_original:
                            found = True
                            break
                    if found:
                        continue
                
                # Column is truly missing
                missing_columns.append(col)
            
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} after cleanup in file {file}.")
                print(f"Found columns: {df_columns}")
                continue
                
            # Select the relevant columns
            df_selected = df[columns_to_select]
            
            # Filter out rows where ZIP is not a 5-character integer
            df_selected = df_selected.copy()  # To avoid SettingWithCopyWarning
            
            # Convert ZIP to string and filter for 5-character integers
            df_selected['ZIP'] = df_selected['ZIP'].astype(str)
            valid_zip_mask = df_selected['ZIP'].str.match(r'^\d{5}$')
            
            # Count filtered rows
            filtered_count = (~valid_zip_mask).sum()
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} rows with invalid ZIP codes.")
            
            df_selected = df_selected[valid_zip_mask]
            total_rows_kept += len(df_selected)
            
            # Rename columns according to mapping
            df_selected = df_selected.rename(columns=col_mapping)
            
            # Add year and append to combined DataFrame
            df_selected['YEAR'] = year
            combined_df = pd.concat([combined_df, df_selected], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            
    print(f"\nData processing summary:")
    print(f"- Total rows processed: {total_rows_processed}")
    print(f"- Rows kept after ZIP validation: {total_rows_kept}")
    print(f"- Rows filtered out: {total_rows_processed - total_rows_kept}")
    print(f"- Data shape: {combined_df.shape}")
    print(f"- Years covered: {sorted(combined_df['YEAR'].unique())}")
            
    return combined_df

if __name__ == "__main__":
    # Set the directory and pattern for the files
    directory = 'insurance/TX_Request/'
    pattern = '*_z.xlsx'
    pattern_2 = '*z_wo.xlsx'
    pattern_3 = '*_z_wo_Q3.xlsx'

    # Create a list of files
    file_list = create_file_list(directory, pattern)
    file_list_2 = create_file_list(directory, pattern_2)
    file_list_3 = create_file_list(directory, pattern_3)
    file_list.extend(file_list_2)
    file_list.extend(file_list_3)

    # For the second occurrence of FIRE, use FIRE.1 as pandas automatically renames duplicate columns
    columns = [
        'ZIP', 'LINE', 'PREMIUM IN FORCE AT END OF QTR', 'EXPOSURE IN FORCE AT END OF QTR ($000)',
        'POLICIES IN FORCE AT END OF QTR', 'TOTAL PAID LOSS', 'FIRE', 'TOTAL PAID CLAIMS', 'FIRE.1'
    ]
    
    df = build_combined_df(file_list, columns)
    df.to_csv('insurance/TX_Request/combined_insurance_data.csv', index=False)