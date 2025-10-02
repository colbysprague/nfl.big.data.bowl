"""
NFL Big Data Bowl 2026 - Utility Functions

This module provides utility functions for processing and visualizing
NFL player tracking data, including trajectory extraction and field visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from const import *


def extract_points(play_group):
    """
    Extract (x, y) coordinate tuples from a player's dataframe.
    
    Args:
        play_group (pd.DataFrame): DataFrame with X and Y columns for a single player
        
    Returns:
        list: List of (x, y) tuples representing the player's trajectory
        
    Example:
        >>> player_data = df[df[NFL_ID] == 12345]
        >>> points = extract_points(player_data)
        >>> # [(x1, y1), (x2, y2), ...]
    """
    points = list(zip(play_group[X], play_group[Y]))
    return points


def get_player_points_array(df, game_id, play_id):
    """
    Extract points for all players in a specific game and play.
    
    This function filters the dataframe to a specific game/play combination,
    groups by player, and extracts their trajectories in frame order.
    
    Args:
        df (pd.DataFrame): DataFrame with player tracking data
        game_id (int): Game identifier
        play_id (int): Play identifier
        
    Returns:
        tuple: A 3-tuple containing:
            - points_array (list): List of lists of (x, y) tuples, one per player
            - labels (list): List of player labels/names
            - ball_land (tuple): Tuple of (x, y) for ball landing, or None if unavailable
            
    Example:
        >>> points, labels, ball_land = get_player_points_array(df, 2023090800, 56)
        >>> print(f"Found {len(points)} players")
        >>> print(f"Ball landed at: {ball_land}")
    """
    # Filter to specific game and play
    play_df = df[(df[GAME_ID] == game_id) & (df[PLAY_ID] == play_id)]
    
    # Sort by frame to ensure correct order
    play_df = play_df.sort_values(FRAME_ID)
    
    points_array = []
    labels = []
    
    # Get ball landing position (should be same for all rows in this play)
    ball_land = None
    if BALL_LAND_X in play_df.columns and BALL_LAND_Y in play_df.columns:
        ball_x = play_df[BALL_LAND_X].iloc[0]
        ball_y = play_df[BALL_LAND_Y].iloc[0]
        if pd.notna(ball_x) and pd.notna(ball_y):
            ball_land = (ball_x, ball_y)
    
    # Group by player (NFL_ID)
    for nfl_id, player_group in play_df.groupby(NFL_ID):
        # Extract points for this player
        points = list(zip(player_group[X], player_group[Y]))
        points_array.append(points)
        
        # Create label (use player name if available, otherwise NFL_ID)
        if PLAYER_NAME in player_group.columns:
            player_name = player_group[PLAYER_NAME].iloc[0]
            labels.append(player_name)
        else:
            labels.append(f'Player {nfl_id}')
    
    return points_array, labels, ball_land


def is_offensive_player(player_role):
    """
    Filter for offensive players based on their role.
    
    Args:
        player_role (pd.Series): Series of player roles
        
    Returns:
        pd.Series: Boolean mask indicating offensive players
        
    Example:
        >>> off_df = df[is_offensive_player(df[PLAYER_ROLE])]
    """
    return player_role.isin(['Other Route Runner', 'Passer', 'Targeted Receiver'])


def plot_multiple_points(points_array, ball_land=None, game_id=None, play_id=None, labels=None):
    """
    Plot multiple player trajectories on a football field visualization.
    
    Creates an interactive football field plot with:
    - Green field background with endzones
    - White yard lines and sidelines
    - Colored player trajectories with arrows showing direction
    - Optional ball landing position marked with a star
    
    Args:
        points_array (list): List of lists of (x, y) tuples - [[points1], [points2], ...]
        ball_land (tuple, optional): Tuple of (x, y) for ball landing position
        game_id (int, optional): Game identifier for title
        play_id (int, optional): Play identifier for title
        labels (list, optional): List of player labels for legend
        
    Example:
        >>> points, labels, ball_land = get_player_points_array(df, game_id, play_id)
        >>> plot_multiple_points(points, ball_land, game_id, play_id, labels)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Draw football field background
    # Main field (green)
    field = patches.Rectangle((0, 0), 120, 53.3, linewidth=2, 
                              edgecolor='white', facecolor='#2d5016', zorder=0)
    ax.add_patch(field)
    
    # End zones (darker green)
    left_endzone = patches.Rectangle((-10, 0), 10, 53.3, linewidth=0, 
                                     facecolor='#1a3d0a', zorder=0)
    right_endzone = patches.Rectangle((120, 0), 10, 53.3, linewidth=0, 
                                      facecolor='#1a3d0a', zorder=0)
    ax.add_patch(left_endzone)
    ax.add_patch(right_endzone)
    
    # Draw yard lines
    for yard in range(0, 121, 10):
        ax.plot([yard, yard], [0, 53.3], 'w-', linewidth=1, alpha=0.5, zorder=1)
    
    # Draw sidelines
    ax.plot([0, 120], [0, 0], 'w-', linewidth=2, zorder=1)
    ax.plot([0, 120], [53.3, 53.3], 'w-', linewidth=2, zorder=1)
    
    # Set limits to show full field with buffer (endzones)
    ax.set_xlim(-10, 130)
    ax.set_ylim(-5, 58.3)
    
    # Plot each trajectory
    colors = plt.cm.tab10(range(len(points_array)))  # Get distinct colors
    
    for i, points in enumerate(points_array):
        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Determine label
        label = labels[i] if labels and i < len(labels) else f'Player {i+1}'
        
        # Plot the trajectory
        ax.plot(x_coords, y_coords, '-o', markersize=5, linewidth=2, 
                color=colors[i], label=label, alpha=0.9, zorder=2)
        
        # Add arrow at the end
        if len(points) >= 2:
            dx = x_coords[-1] - x_coords[-2]
            dy = y_coords[-1] - y_coords[-2]
            ax.arrow(x_coords[-2], y_coords[-2], dx, dy, 
                     head_width=2, head_length=1.5, 
                     fc=colors[i], ec=colors[i], linewidth=2, zorder=3)
    
    # Plot ball landing position
    if ball_land is not None:
        ax.plot(ball_land[0], ball_land[1], marker='*', markersize=20, 
                color='yellow', markeredgecolor='black', markeredgewidth=2,
                label='Ball Landing', zorder=4)
    
    # Labels and grid
    ax.set_xlabel('X (yards)', color='white', fontsize=12)
    ax.set_ylabel('Y (yards)', color='white', fontsize=12)
    
    title = 'Player Trajectories'
    if game_id and play_id:
        title = f'Game: {game_id}, Play: {play_id}'
    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    
    # Style adjustments for football field look
    ax.set_facecolor('#2d5016')
    fig.patch.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.set_aspect('auto')
    
    plt.tight_layout()
    plt.show()


def plot_single_trajectory(points, game_id=None, play_id=None):
    """
    Plot a single player's trajectory on a football field.
    
    Simplified version of plot_multiple_points for visualizing a single player.
    
    Args:
        points (list): List of (x, y) tuples representing trajectory
        game_id (int, optional): Game identifier for title
        play_id (int, optional): Play identifier for title
        
    Example:
        >>> qb_data = df[df[PLAYER_ROLE] == 'Passer']
        >>> points = extract_points(qb_data)
        >>> plot_single_trajectory(points, game_id, play_id)
    """
    # Extract x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Set limits to show full field with buffer
    plt.xlim(-5, 125)
    plt.ylim(-5, 58)
    
    # Plot the points
    plt.plot(x_coords, y_coords, 'b-o', markersize=4, linewidth=1)
    
    # Mark start and end
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    # Labels and grid
    plt.xlabel('X (yards)')
    plt.ylabel('Y (yards)')
    
    title = 'Player Trajectory'
    if game_id and play_id:
        title = f'Game: {game_id}, Play: {play_id}'
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().set_aspect('auto')
    
    plt.show()