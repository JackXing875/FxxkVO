import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryVisualizer3D:
    """Provides real-time 3D visualization for camera trajectories.

    This class manages a Matplotlib-based 3D canvas to render the estimated 
    camera path dynamically. It supports interactive manipulation and 
    automated high-resolution exports.
    """

    def __init__(self):
        """Initializes the 3D visualization canvas and rendering parameters."""
        # Enable Matplotlib interactive mode for non-blocking updates.
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Configure industrial dark-themed UI aesthetics.
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')
        
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        self.ax.set_title("DeepVO 3D Trajectory (Interactive)", color='white', pad=20)
        self.ax.set_xlabel("X (Lateral)")
        self.ax.set_ylabel("Z (Longitudinal)")
        self.ax.set_zlabel("Y (Vertical)")
        
        # Containers for historical trajectory coordinates.
        self.xs = []
        self.ys = []
        self.zs = []
        
        # Initialize the primary trajectory line object.
        self.line, = self.ax.plot([], [], [], color='#ff0055', linewidth=2, label="Estimated Path")
        
        # Mark the coordinate origin as the sequence starting point.
        self.ax.scatter([0], [0], [0], color='cyan', marker='o', s=30, label="Origin")
        
        legend = self.ax.legend(facecolor='#2b2b2b', edgecolor='white')
        for text in legend.get_texts():
            text.set_color("white")

    def update(self, x, y, z):
        """Updates the 3D plot with a new camera pose coordinate.

        Args:
            x (float): The current X-coordinate (lateral displacement).
            y (float): The current Y-coordinate (vertical displacement).
            z (float): The current Z-coordinate (forward displacement).
        """
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        
        # Map physical VO coordinates to visualization axes.
        # Note: VO Z (forward) maps to plot Y; VO Y (up/down) maps to plot Z.
        self.line.set_data(self.xs, self.zs) 
        self.line.set_3d_properties(self.ys)
        
        # Perform dynamic axis scaling to keep the trajectory centered in view.
        margin = 2.0
        self.ax.set_xlim(min(self.xs) - margin, max(self.xs) + margin)
        self.ax.set_ylim(min(self.zs) - margin, max(self.zs) + margin) 
        self.ax.set_zlim(min(self.ys) - margin, max(self.ys) + margin) 
        
        # Trigger GUI event loop processing to handle rendering and user interactions.
        plt.pause(0.001)

    def close(self, save_path=None):
        """Finalizes the visualization session and handles figure export.

        Args:
            save_path (str, optional): The file system path to save the high-resolution 
                3D trajectory plot. Defaults to None.
        """
        plt.ioff()
        
        # Export the final trajectory plot as a high-resolution image if requested.
        if save_path:
            self.fig.savefig(
                save_path, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor=self.fig.get_facecolor()
            )
            print(f"High-resolution 3D trajectory capture saved to: {save_path}")
            
        plt.show()