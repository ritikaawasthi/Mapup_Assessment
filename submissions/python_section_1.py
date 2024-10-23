from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    
    Parameters:
    lst (List[int]): List of integers to be reversed in groups.
    n (int): Size of the groups for reversing.
    
    Returns:
    List[int]: List with every group of n elements reversed.
    """
    result = []  # To store the final result
    length = len(lst)
    
    # Iterate through the list in steps of n
    for i in range(0, length, n):
        # Create a temporary list to hold the current group
        group = []
        
        # Collect elements for the current group
        for j in range(i, min(i + n, length)):
            group.append(lst[j])
        
        # Manually reverse the current group and add to the result
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    
    return result
if __name__ == "__main__":
    # Prompt the user for a list of integers
    user_input = input("Enter a list of integers separated by spaces: ")
    lst = list(map(int, user_input.split()))  # Convert input to a list of integers
    
    # Prompt the user for the group size
    n = int(input("Enter the group size for reversing: "))
    
    # Get the reversed list
    output = reverse_by_n_elements(lst, n)
    
    # Print the output
    print("Reversed list:", output)



from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary where:
    
    - The keys are the lengths of the strings.
    - The values are lists of strings that have the same length as the key.
    
    Parameters:
    lst (List[str]): List of strings to be grouped by their length.
    
    Returns:
    Dict[int, List[str]]: Dictionary where keys are string lengths and values are lists of strings with that length.
    """
    length_dict = {}  # Initialize an empty dictionary to store the result
    
    for string in lst:
        length = len(string)  # Calculate the length of the current string
        
        # Check if the length is already a key in the dictionary
        if length not in length_dict:
            length_dict[length] = []  # If not, create a new list for this length
        
        # Add the string to the corresponding list
        length_dict[length].append(string)
    
    # Sort the dictionary by key (string length) and return it
    return dict(sorted(length_dict.items()))

if __name__ == "__main__":
    # Example usage
    strings = input("Enter a list of strings separated by spaces: ").split()
    grouped = group_by_length(strings)
    print("Grouped by length:", grouped)



from typing import Dict, Any

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened_dict = {}

    def flatten_helper(sub_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in sub_dict.items():
            # Create the new key by appending the current key to the parent key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten the nested dictionary
                flatten_helper(value, new_key)
            elif isinstance(value, list):
                # Handle lists by indexing
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten_helper(item, f"{new_key}[{i}]")
                    else:
                        flattened_dict[f"{new_key}[{i}]"] = item
            else:
                # Base case for non-dictionary and non-list values
                flattened_dict[new_key] = value

    flatten_helper(nested_dict)
    return flattened_dict

# Example usage
nested_dict_example = {
    'name': 'John',
    'info': {
        'age': 30,
        'contacts': {
            'email': 'john@example.com',
            'phone': '123-456-7890'
        },
        'addresses': [
            {'type': 'home', 'address': '123 Main St'},
            {'type': 'work', 'address': '456 Elm St'}
        ]
    },
    'tags': ['friend', 'developer']
}

flattened = flatten_dict(nested_dict_example)
print(flattened)



from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start, end):
        if start == end:
            result.append(nums[:])  # Append a copy of the current permutation
        for i in range(start, end):
            # Skip generating permutations for duplicates
            if i > start and nums[i] == nums[start]:
                continue
            # Swap elements to generate the next permutation
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1, end)
            # Swap back to restore the original list order
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()  # Sort the list to ensure that duplicates are adjacent
    backtrack(0, len(nums))
    return result

# Example usage
input_list = [1, 1, 2]
unique_perms = unique_permutations(input_list)
print(unique_perms)


import pandas as pd
import polyline
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate distance between two lat/long points
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coords = polyline.decode(polyline_str)
    
    # Create a DataFrame with latitude and longitude columns
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with 0 for the first point
    df['distance'] = 0.0
    
    # Iterate over the DataFrame and calculate distances between consecutive points
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        
        # Calculate the distance using the Haversine formula
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

# Example usage
polyline_str = "u{~vFvyys@fEgD"
df = polyline_to_dataframe(polyline_str)
print(df)


from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all other elements in the same row and column.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Step 1: Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: For each element, replace it with the sum of the elements in the same row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate sum of the row excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            # Calculate sum of the column excluding the current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            # Store the result in the final matrix
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)

# Output the transformed matrix
for row in result:
    print(row)


import pandas as pd
from datetime import time

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verifies if each (id, id_2) pair in the dataframe has timestamps that cover a full 24-hour period and span all 7 days.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'.
    
    Returns:
        pd.Series: Boolean series with multi-index (id, id_2) indicating if the timestamps are incorrect (True) or correct (False).
    """
    
    # Define the full set of days of the week and the time span for completeness check
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    full_day_timespan = (time(0, 0, 0), time(23, 59, 59))  # 24-hour period from midnight to end of the day
    
    # Function to check for full day and time coverage for a (id, id_2) group
    def is_incomplete(group):
        # Drop the grouping columns if they are passed into the group
        group = group.drop(columns=['id', 'id_2'])
        
        # Collect the unique days for the group
        covered_days = set(group['startDay'].unique())
        
        # Debug print to check days covered
        print(f"Checking group (id, id_2)")
        print(f"Covered days: {covered_days}")
        
        # Check if all 7 days of the week are covered
        if set(days_of_week) != covered_days:
            print("Missing days detected!")
            return True
        
        # For each day, check if it spans the full 24 hours (by analyzing startTime and endTime)
        for day in days_of_week:
            day_data = group[group['startDay'] == day]
            
            # Sort by startTime for that day
            day_data_sorted = day_data.sort_values(by='startTime')
            
            # Track the current time to check gaps
            current_time = full_day_timespan[0]  # start at 00:00:00
            
            # Debug print to check sorted times for the day
            print(f"Checking time coverage for {day}")
            print(day_data_sorted[['startTime', 'endTime']])
            
            for _, row in day_data_sorted.iterrows():
                # Check for any gap in the time span
                if current_time < row['startTime']:
                    print(f"Time gap detected! Current time: {current_time}, Next start time: {row['startTime']}")
                    return True  # Incomplete time coverage if there's a gap
                
                # Update current_time to the end of the current time span
                current_time = row['endTime']
            
            # Ensure the final time of the day reaches 23:59:59
            if current_time < full_day_timespan[1]:
                print(f"Day not fully covered! Final time: {current_time}")
                return True  # The day is not fully covered
        
        return False  # All days and times are covered correctly
    
    # Apply the check for each (id, id_2) group
    grouped = df.groupby(['id', 'id_2']).apply(is_incomplete)
    
    return grouped



# Example usage:
# Assuming 'dataset-1.csv' is the path to your dataset
df = pd.read_csv(r'C:\Users\DELL\Desktop\dataset-1.csv')

# Convert startTime and endTime to datetime.time objects if they are not already
df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

result = time_check(df)
print(result)
