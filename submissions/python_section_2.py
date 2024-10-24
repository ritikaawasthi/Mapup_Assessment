import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Extract unique toll locations (IDs)
    toll_locations = pd.concat([df['id_start'], df['id_end']]).unique()
    
    # Initialize the distance matrix with infinity for non-diagonal and 0 for diagonal
    dist_matrix = pd.DataFrame(np.inf, index=toll_locations, columns=toll_locations)
    
    # Set the diagonal to 0 (distance from a location to itself)
    np.fill_diagonal(dist_matrix.values, 0)

    # Fill in the direct distances from the dataframe
    for _, row in df.iterrows():
        ID1, ID2, distance = row['id_start'], row['id_end'], row['distance']
        dist_matrix.loc[ID1, ID2] = distance
        dist_matrix.loc[ID2, ID1] = distance  # Ensure the matrix is symmetric
    
   # compute the shortest paths between all locations
    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                dist_matrix.loc[i, j] = min(dist_matrix.loc[i, j], dist_matrix.loc[i, k] + dist_matrix.loc[k, j])
    
    return dist_matrix

if __name__ == "__main__":
    # Load the dataset (dataset-2.csv)
    df = pd.read_csv(r'C:\Users\DELL\Desktop\dataset-2.csv')
    
    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(df)
    
    # Print the result
    print(distance_matrix)


def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix with toll locations as index and columns.

    Returns:
        pd.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to collect rows
    unrolled_data = []

    # Iterate over the matrix, avoiding diagonal elements
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Skip diagonal (same id_start and id_end)
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])
    
    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df

if __name__ == "__main__":

    # Call the 'calculate_distance_matrix' to generate the distance matrix
    
    df = pd.read_csv(r'C:\Users\DELL\Desktop\dataset-2.csv')
    distance_matrix = calculate_distance_matrix(df)
    
    # Step 2: Unroll the distance matrix using the unroll_distance_matrix function
    unrolled_df = unroll_distance_matrix(distance_matrix)
    
    # Print the unrolled DataFrame
    print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: str) -> list:
    """
    Find ids from the DataFrame where their average distances lie within 10% 
    of the reference id's average distance.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (str): The reference id_start value to compare against.

    Returns:
        list: Sorted list of id_start values that lie within 10% of the reference id's average distance.
    """
    # Ensure the id_start column is treated as strings
    df['id_start'] = df['id_start'].astype(str)
    
    # Calculate the average distance for the reference_id
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    print(f"Reference ID: {reference_id}, Average Distance: {reference_avg_distance}")
    
    # Ensure there's a valid reference average distance
    if pd.isna(reference_avg_distance):
        print(f"No distances found for reference ID {reference_id}.")
        return []
    
    # Calculate the 10% threshold
    threshold_lower = reference_avg_distance * 0.9
    threshold_upper = reference_avg_distance * 1.1
    print(f"Threshold Range: {threshold_lower} - {threshold_upper}")
    
    # Find the average distance for each id_start
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    print("\nAverage distances for all IDs:")
    print(avg_distances)
    
    # Filter ids within the 10% range of the reference average distance
    filtered_ids = avg_distances[(avg_distances['distance'] >= threshold_lower) &
                                 (avg_distances['distance'] <= threshold_upper)]
    
    # Exclude the reference ID and sort
    result_ids = sorted(filtered_ids[filtered_ids['id_start'] != reference_id]['id_start'].tolist())
    
    print(f"\nFiltered IDs within 10% threshold: {result_ids}")
    return result_ids

if __name__ == "__main__":
    # Sample DataFrame from Question 10 (for testing)
    unrolled_df = pd.DataFrame({
        'id_start': ['1001400', '1001400', '1001400', '1001402', '1001402', '1001404'],
        'id_end': ['1001402', '1001404', '1001406', '1001400', '1001404', '1001400'],
        'distance': [9.7, 29.9, 15.2, 9.7, 21.4, 29.9]
    })
    
    # Example usage: finding IDs within 10% threshold of reference id 1001400
    reference_id = '1001400'  # Pass reference ID as string
    ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    
    # Print the result
    print(f"IDs within 10% threshold of average distance for ID {reference_id}: {ids_within_threshold}")




def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates based on vehicle types and distances.

    Args:
        df (pd.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance'.

    Returns:
        pd.DataFrame: Updated DataFrame with additional columns for toll rates.
    """
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add as new columns
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df

if __name__ == "__main__":
    
    unrolled_df = unroll_distance_matrix(distance_matrix)  
    
    # Calculate the toll rates
    toll_rate_df = calculate_toll_rate(unrolled_df)
    
    # Print the result
    print(toll_rate_df)


from datetime import time

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates based on vehicle types for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing distance data.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated toll rates for each vehicle type.
    """
    # Rate coefficients
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pd.DataFrame): DataFrame containing vehicle toll rates.

    Returns:
        pd.DataFrame: Updated DataFrame with time-based toll rates and time columns.
    """
    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Initialize lists to store new data
    new_data = []
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        # Cover a full 24-hour period for each day of the week
        for day in days_of_week:
            for hour in range(24):
                # Determine start and end time
                start_time = time(hour, 0)  # Start of the hour
                end_time = time(hour, 59, 59)  # End of the hour
                
                # Calculate discount factor based on time
                if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    # Weekdays
                    if hour < 10:
                        discount_factor = 0.8
                    elif 10 <= hour < 18:
                        discount_factor = 1.2
                    else:
                        discount_factor = 0.8
                else:
                    # Weekends
                    discount_factor = 0.7
                
                # Calculate new toll rates applying the discount factor
                updated_rates = {vehicle: row[vehicle] * discount_factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                
                # Create a new row with calculated values
                new_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': row['distance'],
                    'moto': updated_rates['moto'],
                    'car': updated_rates['car'],
                    'rv': updated_rates['rv'],
                    'bus': updated_rates['bus'],
                    'truck': updated_rates['truck'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time
                })
    
    # Create a new DataFrame from the new data
    time_based_df = pd.DataFrame(new_data)
    
    return time_based_df

if __name__ == "__main__":
    
    unrolled_df = pd.DataFrame({
        'id_start': ['1001400', '1001400', '1001400', '1001402', '1001402', '1001404'],
        'id_end': ['1001402', '1001404', '1001406', '1001400', '1001404', '1001400'],
        'distance': [9.7, 29.9, 15.2, 9.7, 21.4, 29.9]
    })

    # Calculate toll rates
    toll_rate_df = calculate_toll_rate(unrolled_df)

    # Calculate time-based toll rates
    time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)

    # Print the result
    print(time_based_toll_df)






