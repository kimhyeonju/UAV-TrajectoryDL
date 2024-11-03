import pandas as pd

# Load the first file
file1_path = '/GP_final_second/data/aircraft/bi_gru_model_v3/aircraft/look_back=10&forward=0/training_log_general_v3.txt'
with open(file1_path, 'r') as f:
    log1 = f.readlines()

# Load the second file
file2_path = '/Users/admin/PycharmProjects/GP_test/GP_final/data/aircraft/look_back=10&forward=0/training_log_general.txt'
with open(file2_path, 'r') as f:
    log2 = f.readlines()


# Function to extract MSE values from log data
def extract_mse(log_lines):
    train_mse = []
    test_mse = []
    file_names = []

    for line in log_lines:
        if 'Processing file' in line:
            file_name = line.split(': ')[1].strip()
            file_names.append(file_name)
        if 'Train MSE' in line:
            train_value = float(line.split(': ')[1].strip())
            train_mse.append(train_value)
        if 'Test MSE' in line:
            test_value = float(line.split(': ')[1].split(',')[0].strip())
            test_mse.append(test_value)

    return pd.DataFrame({'File': file_names, 'Train MSE': train_mse, 'Test MSE': test_mse})


# Extract MSE values from both logs
df1 = extract_mse(log1)
df2 = extract_mse(log2)

# Merge both dataframes on the 'File' column to compare MSE values
comparison_df = pd.merge(df1, df2, on='File', suffixes=('_v3', '_v4'))

# Display the comparison dataframe
# import ace_tools as tools;

print(comparison_df)