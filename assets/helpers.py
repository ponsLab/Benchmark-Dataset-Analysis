from enum import IntEnum
import seaborn as sns

class Activity(IntEnum):
    WALKING     = 1
    RAMPS_UP    = 2
    RAMPS_DOWN  = 3
    STAIRS_UP   = 4
    STAIRS_DOWN = 5
    UNDEFINED   = 6

class GaitState(IntEnum):
    EARLY_STANCE = 0
    LATE_STANCE = 1
    EARLY_SWING = 2
    LATE_SWING = 3
    UNDEFINED = 4

class Leg(IntEnum):
    LEFT = 0
    RIGHT = 1

class JointName(IntEnum):
    LEFT_HIP = 0
    LEFT_KNEE = 1
    RIGHT_HIP = 2
    RIGHT_KNEE = 3
    BACKPACK  = 4
 
# Used to map code (numbers) to names (string)
condition_mapping = {
    'transparent_UNDEFINED': 0,
    'transparent_WALKING': 1,
    'SM_UNDEFINED': 0,
    'SM_WALKING': 1,
    'transparent_RAMPS_UP': 2,
    'transparent_RAMPS_DOWN': 3,
    'SM_RAMPS_DOWN': 3,
    'SM_RAMPS_UP': 2,
    'SM_STAIRS_UP': 4,
    'SM_STAIRS_DOWN': 5,
    'transparent_STAIRS_UP': 4,
    'transparent_STAIRS_DOWN': 5
}

# Color /ordering for plots
order = ['transparent_WALKING','SM_WALKING','transparent_RAMPS_UP','SM_RAMPS_UP','transparent_RAMPS_DOWN','SM_RAMPS_DOWN',
         'transparent_STAIRS_UP','SM_STAIRS_UP','transparent_STAIRS_DOWN','SM_STAIRS_DOWN']
light_palette = sns.color_palette("pastel", 5)
dark_palette = sns.color_palette("dark", 5)
custom_palette = [color for pair in zip(light_palette, dark_palette) for color in pair]

# Define color mapping for SM and transparent conditions
color_mapping = {
    'transparent': 'tab:blue',
    'SM': 'tab:orange'
}

def format_activity(activity):
    """
    Converts an activity string into a more readable format.

    Args:
        activity: The activity string to be formatted (e.g., "WALKING", "STAIRS_UP").

    Returns:
        The formatted activity string with spaces replacing underscores and the first letter capitalized.
    """

    # Replace underscores with spaces
    formatted_activity = activity.replace("_", " ")

    # Capitalize the first letter of each word
    formatted_activity = formatted_activity.title()

    return formatted_activity


import pandas as pd

def check_transition(ankle_position_actual, condition):
    """
    Checks if a transition from states 1, 2, or 3 to state 0 occurs,
    or if the condition changes from UNDEFINED to a defined state.

    Args:
        ankle_position_actual: Pandas Series representing the ankle position.
        condition: Pandas Series representing the condition.

    Returns:
        Pandas Series of booleans indicating transitions.
    """
    return (
        (ankle_position_actual == 0) & (ankle_position_actual.shift(1).isin())
    ) | (condition.shift(1).str.contains("UNDEFINED") & ~condition.str.contains("UNDEFINED"))


def assign_step_number(transition, step_number):
    """
    Assigns step numbers based on transitions.

    Args:
        transition: Pandas Series of booleans indicating transitions.
        step_number:  Integer representing the current step number.

    Returns:
        Pandas Series of step numbers.
    """
    return transition.cumsum() + step_number


def check_step_completeness(df, step_number_col, ankle_position_col):
    """
    Checks the completeness of steps based on the number of unique ankle positions.

    Args:
        df: Pandas DataFrame containing the data.
        step_number_col: String representing the column name for step numbers.
        ankle_position_col: String representing the column name for ankle positions.

    Returns:
        Pandas Series of booleans indicating step completeness.
    """
    df['step_complete'] = True
    for step in df[step_number_col].unique():
        if df[df[step_number_col] == step][ankle_position_col].nunique() < 3:
            df.loc[df[step_number_col] == step, "step_complete"] = False
        elif (df[df[step_number_col] == step][ankle_position_col].nunique() == 3):
            if ~(df[df[step_number_col] == step][ankle_position_col].unique() == [1., 2., 3.]).all() | ~("UNDEFINED" in df.iloc[df[(df[step_number_col] == 1)].iloc.name - 1].condition):
                df.loc[df[step_number_col] == step, "step_complete"] = False
    return df['step_complete']


def process_steps(df, ankle_position_col_left, ankle_position_col_right, condition_col):
    """
    Processes step data to identify transitions, assign step numbers, and check completeness.

    Args:
        df: Pandas DataFrame containing the data.
        ankle_position_col_left: String representing the column name for left ankle positions.
        ankle_position_col_right: String representing the column name for right ankle positions.
        condition_col: String representing the column name for the condition.

    Returns:
        Pandas DataFrame with added columns for step information.
    """

    df['transition_left'] = check_transition(df[ankle_position_col_left], df[condition_col])
    df['transition_right'] = check_transition(df[ankle_position_col_right], df[condition_col])

    df['step_number_l'] = assign_step_number(df['transition_left'], 0)
    df['step_number_r'] = assign_step_number(df['transition_right'], 0)

    df['step_complete_l'] = check_step_completeness(df.copy(), 'step_number_l', ankle_position_col_left)
    df['step_complete_r'] = check_step_completeness(df.copy(), 'step_number_r', ankle_position_col_right)

    return df