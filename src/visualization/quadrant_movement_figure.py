import matplotlib.pyplot as plt
import pandas as pd

def generate_quadrant_movement_figure():

    # Data as a CSV-like format
    data = {
        'Stereotype': ['abstract', 'category', 'collective', 'datatype', 'enumeration', 'event', 'historicalRole',
                       'historicalRoleMixin', 'kind', 'mixin', 'mode', 'phase', 'phaseMixin', 'quality'],
        'quadrant_start': ['Q3', 'Q1', 'Q1', 'Q3', 'Q3', 'Q1', 'Q3', 'Q3', 'Q1', 'Q2', 'Q1', 'Q1', 'Q3', 'Q3'],
        'quadrant_end': ['Q3', 'Q1', 'Q2', 'Q1', 'Q3', 'Q1', 'Q3', 'Q3', 'Q1', 'Q3', 'Q1', 'Q1', 'Q3', 'Q1']
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Filter elements that moved (start != end)
    df_moved = df[df['quadrant_start'] != df['quadrant_end']]

    # Filter elements that stayed in their quadrants (start == end)
    df_static = df[df['quadrant_start'] == df['quadrant_end']]

    # Define the positions of the quadrants in a cross alignment
    quadrant_positions = {
        'Q1': (0, 1),  # Top
        'Q2': (-1, 0),  # Left-middle
        'Q3': (0, -1),  # Bottom
        'Q4': (1, 0)  # Right-middle
    }

    # Offset for starting and ending arrows (to avoid overlap with quadrant labels)
    arrow_offset = 0.2

    # Adjusted positions for the arrows to start and end outside quadrant labels
    adjusted_positions = {
        'Q1': (0, 1 - arrow_offset),
        'Q2': (-1 + arrow_offset, 0),
        'Q3': (0, -1 + arrow_offset),
        'Q4': (1 - arrow_offset, 0)
    }

    # Group by start and end quadrant to combine labels for the elements that moved
    grouped_moved = df_moved.groupby(['quadrant_start', 'quadrant_end'])['Stereotype'].apply(list).reset_index()

    # Group by quadrant for elements that did not move
    grouped_static = df_static.groupby('quadrant_start')['Stereotype'].apply(list).reset_index()

    # Create a figure and axis with a more aesthetically pleasing size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the quadrants with enhanced visual style
    for quadrant, pos in quadrant_positions.items():
        ax.text(pos[0], pos[1], quadrant, fontsize=18, ha='center', va='center', color='navy', fontweight='bold',
                bbox=dict(facecolor='lightblue', alpha=0.6, edgecolor='navy', boxstyle='round,pad=0.4'))

    # Plot combined arrows and labels with improved arrow style and label visibility
    for index, row in grouped_moved.iterrows():
        start_pos = adjusted_positions[row['quadrant_start']]
        end_pos = adjusted_positions[row['quadrant_end']]

        # Draw an elegant arrow between start and end
        ax.annotate(
            '', xy=end_pos, xytext=start_pos,
            arrowprops=dict(facecolor='coral', edgecolor='coral', shrink=0.05, width=1.5, headwidth=10, headlength=10)
        )

        # Label the arrow with the combined Stereotype names, with improved aesthetics
        midpoint = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
        label = ', '.join(row['Stereotype'])
        ax.text(midpoint[0], midpoint[1], label, fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    # Plot the elements that did not move near their respective quadrants in a single box
    # Combine the elements for each quadrant into a single box
    for index, row in grouped_static.iterrows():
        quadrant = row['quadrant_start']

        # Combine the stereotypes into a single string
        combined_elements = ', '.join(row['Stereotype'])

        # Offset positions for the combined box
        offset_positions = {
            'Q1': (0, 1 + 0.2),
            'Q2': (-1 - 0.3, 0),
            'Q3': (0, -1 - 0.2),
            'Q4': (1 + 0.3, 0)
        }
        text_position = offset_positions[quadrant]

        # Plot the combined box for static elements in the respective position
        ax.text(text_position[0], text_position[1], combined_elements, fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='lightgray', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set limits and labels with a more polished appearance
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a grid for aesthetics and better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Add title with improved aesthetics
    plt.title('Quadrant Movements of Stereotypes', fontsize=16, fontweight='bold',
              color='darkslategray')

    # Adjust the aspect ratio for a balanced look
    ax.set_aspect('equal')

    # Show the aesthetically enhanced plot
    plt.show()
