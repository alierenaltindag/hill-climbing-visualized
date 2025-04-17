import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import random
import re

# For theming - ensure ttkthemes is installed (pip install ttkthemes)
try:
    from ttkthemes import ThemedTk
    HAS_THEMED_TK = True
except ImportError:
    print("ttkthemes not found. Using default Tkinter theme. Install with: pip install ttkthemes")
    HAS_THEMED_TK = False

# Define multiple objective functions
def objective_function_single_peak(x):
    """
    A 1D objective function with a single peak (using Gaussian function).
    """
    # Single peak (simple case for basic demonstration)
    x1, sigma1, amp1 = 0, 1.0, 2.0
    peak = amp1 * np.exp(-((x - x1)**2) / (2 * sigma1**2))
    
    return peak

def objective_function_two_peaks(x):
    """
    A 1D objective function with two peaks (using Gaussian functions).
    """
    # Peak 1
    x1, sigma1, amp1 = -3, 0.8, 1.5
    peak1 = amp1 * np.exp(-((x - x1)**2) / (2 * sigma1**2))

    # Peak 2
    x2, sigma2, amp2 = 2, 1.2, 1.8
    peak2 = amp2 * np.exp(-((x - x2)**2) / (2 * sigma2**2))

    return peak1 + peak2

def objective_function_long_straights(x):
    """
    A 1D objective function with long straight segments with slight slopes.
    Shows how small step sizes can make convergence very slow on gradual slopes.
    """
    # Base function with slight upward slopes instead of flat plateaus
    if x < -4:
        # Downward slope in far left region
        base = -0.3 * x - 0.5
    elif -4 <= x <= -2:
        # Slight upward slope in first "plateau" region
        base = 0.05 * x + 1.2
    elif -2 < x < 1:
        # Steeper connecting region
        base = 0.3 * x + 0.7
    elif 1 <= x <= 3:
        # Slight upward slope in second "plateau" region
        base = 0.08 * x + 1.9
    else:
        # Downward slope in far right region
        base = -0.3 * x + 3.0
    
    # Add smooth transition at the center
    center_bump = 0.5 * np.exp(-((x - 0)**2) / 0.3)
    
    return base + center_bump

def objective_function_complex(x):
    """
    A complex 1D objective function with 3 peaks, including two close peaks
    where one is local maximum and another is global maximum.
    """
    # First peak (medium)
    x1, sigma1, amp1 = -3.5, 0.7, 1.8
    peak1 = amp1 * np.exp(-((x - x1)**2) / (2 * sigma1**2))
    
    # Second peak (local maximum, part of close pair)
    x2, sigma2, amp2 = 1.2, 0.4, 2.0
    peak2 = amp2 * np.exp(-((x - x2)**2) / (2 * sigma2**2))
    
    # Third peak (global maximum, close to second peak)
    x3, sigma3, amp3 = 2.0, 0.3, 2.3
    peak3 = amp3 * np.exp(-((x - x3)**2) / (2 * sigma3**2))
    
    # Add slight slope
    slope = 0.1 * x
    
    # Add small perturbations
    noise = 0.05 * np.sin(5 * x)
    
    return peak1 + peak2 + peak3 + slope + noise

def objective_function_many_local_optima(x):
    """
    A function with many local optima to show how easily hill climbing gets stuck.
    This demonstrates the need for multiple restarts with different initial positions.
    """
    # Base curve
    base = 2 * np.exp(-((x - 0)**2) / 8)
    
    # Add many local maxima using sine waves with increasing frequency
    local_maxima = 0.5 * np.sin(5 * x) + 0.3 * np.sin(8 * x) + 0.2 * np.sin(12 * x) + 0.1 * np.sin(20 * x)
    
    # Global optimum near x=0 but with many surrounding local optima
    return base + local_maxima

def objective_function_narrow_peak(x):
    """
    A function with a narrow, tall peak among broader hills.
    Shows how step size needs to be carefully chosen - too large will step over the peak,
    too small will take forever to find it from a distance.
    """
    # Broad hills
    broad_hills = 1.5 * np.exp(-((x + 3)**2) / 4) + 1.2 * np.exp(-((x - 3)**2) / 5)
    
    # Narrow, tall peak that's easy to miss with large step size
    narrow_peak = 2.5 * np.exp(-((x - 0.5)**2) / 0.1)
    
    return broad_hills + narrow_peak

def objective_function_deceptive(x):
    """
    A deceptive function with a gradient leading to a local optimum,
    but a much better global optimum is available in a different direction.
    Shows how hill climbing follows the gradient and can miss better solutions.
    """
    # Gradual slope leading to decent local optimum
    local_path = 0.15 * x + 1.5
    
    # Local hill
    local_hill = 1.0 * np.exp(-((x - 4)**2) / 1.0)
    
    # Hidden global optimum in the opposite direction
    global_hill = 2.5 * np.exp(-((x + 3)**2) / 0.8)
    
    # Overall function has a local optimum that's easier to reach following the gradient,
    # but a better global optimum exists in a different direction
    if x >= 0:
        return local_path + local_hill
    else:
        # Slight downward slope before the global optimum to make it deceptive
        connecting_region = -0.05 * x + 1.5
        return connecting_region + global_hill

def objective_function_plateau_and_cliff(x):
    """
    A function with a wide plateau followed by a steep climb to the optimum.
    Shows how hill climbing struggles in flat regions but converges quickly on clear gradients.
    Demonstrates why step size choice is critical for navigating different landscapes.
    """
    # Nearly flat plateau region with very slight gradient
    if x < 0:
        # Almost flat plateau with barely perceptible slope
        plateau = 0.01 * x + 1.0
    elif 0 <= x < 2:
        # Slightly steeper but still gradual slope
        plateau = 0.05 * x + 1.0
    else:
        # Steep climb to optimum
        plateau = 0.8 * x - 0.5
    
    # Add small local variations on the plateau to create some texture
    texture = 0.03 * np.sin(3 * x)
    
    # Add peak at far right
    peak = 1.5 * np.exp(-((x - 4)**2) / 0.5)
    
    return plateau + texture + peak

# Default function to use initially
objective_function = objective_function_single_peak

class HillClimbingVisualizer:
    def __init__(self, root, max_iterations=30, num_neighbors=5, step_size=0.5):
        self.root = root
        self.root.title("Hill Climbing Algorithm Visualization")
        self.root.geometry("1000x700")
        
        # Bind Ctrl+C to exit the application
        self.root.bind("<Control-c>", self.exit_application)
        
        # Also bind Ctrl+q as an alternative exit shortcut
        self.root.bind("<Control-q>", self.exit_application)
        
        # Set theme-compatible background
        if HAS_THEMED_TK:
            bg_color = self.root.cget('background')  # Get theme background color
        else:
            bg_color = "#f0f0f0"  # Default light gray
        
        self.root.configure(bg=bg_color)
        
        # Set available objective functions
        self.objective_functions = {
            "Single Peak (Default)": objective_function_single_peak,
            "Two Peaks": objective_function_two_peaks,
            "Long Straights": objective_function_long_straights,
            "Complex (3 Peaks)": objective_function_complex,
            "Many Local Optima": objective_function_many_local_optima,
            "Narrow Peak": objective_function_narrow_peak,
            "Deceptive Gradient": objective_function_deceptive,
            "Plateau & Cliff": objective_function_plateau_and_cliff
        }
        self.current_objective_function = objective_function_single_peak
        
        # Algorithm parameters
        self.max_iterations = max_iterations
        self.num_neighbors = num_neighbors
        self.step_size = step_size
        self.algorithm_finished = False
        
        # Animation state
        self.current_iteration = 0
        self.current_x = random.uniform(-6, 6)  # Random starting point
        self.neighbors = []
        self.best_neighbor = None
        self.animation_paused = True
        self.animation = None
        self.anim_speed = 1.0  # Animation speed multiplier (1.0 = normal speed)
        self.base_interval = 500  # Base interval in milliseconds for animation
        self.stop_reason = None  # Track why the algorithm stopped
        
        # Animation stages
        self.STAGE_GENERATE_NEIGHBORS = 0
        self.STAGE_FIND_BEST = 1
        self.STAGE_TRANSITION = 2
        self.animation_stage = self.STAGE_GENERATE_NEIGHBORS
        self.neighbors_to_remove = []
        self.transition_steps = 1
        self.transition_current_step = 0
        
        # Initialize visualization elements
        self.neighbor_area = None
        self.neighbor_points = []
        self.neighbor_lines = []
        self.neighbor_annotations = []
        self.best_neighbor_point = None
        self.best_neighbor_line = None
        self.current_annotation = None
        self.optimum_annotation = None
        
        # Track history for visualization
        self.history_x = [self.current_x]
        self.history_y = [self.current_objective_function(self.current_x)]
        
        # Create main frame with proper styling
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel
        self.create_control_panel()
        
        # Create plot
        self.create_plot()
        
        # Error handling
        self.validation_errors = []
        
    def create_control_panel(self):
        # Control frame with theme styling
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Get background color for consistency
        if HAS_THEMED_TK:
            bg_color = self.root.cget('background')
        else:
            bg_color = "#f0f0f0"
        
        # Title label with styling
        title_label = ttk.Label(control_frame, text="Hill Climbing Algorithm", 
                             font=("Arial", 16, "bold"))
        title_label.pack(pady=5)
        
        # Function selection frame
        func_frame = ttk.Frame(control_frame)
        func_frame.pack(pady=5)
        
        # Function selector
        ttk.Label(func_frame, text="Objective Function:", 
                font=("Arial", 10)).grid(row=0, column=0, padx=5, pady=5)
        
        self.function_var = tk.StringVar(value="Single Peak (Default)")
        function_dropdown = ttk.Combobox(func_frame, textvariable=self.function_var, 
                                      values=list(self.objective_functions.keys()),
                                      width=20, state="readonly")
        function_dropdown.grid(row=0, column=1, padx=5, pady=5)
        function_dropdown.bind("<<ComboboxSelected>>", self.change_objective_function)
        
        # Parameters frame
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(pady=10)
        
        # Max iterations
        ttk.Label(param_frame, text="Max Iterations:").grid(row=0, column=0, padx=5, pady=5)
        self.max_iter_var = tk.IntVar(value=self.max_iterations)
        max_iter_entry = ttk.Entry(param_frame, textvariable=self.max_iter_var, width=5)
        max_iter_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Number of neighbors
        ttk.Label(param_frame, text="Number of Neighbors:").grid(row=0, column=2, padx=5, pady=5)
        self.num_neighbors_var = tk.IntVar(value=self.num_neighbors)
        num_neighbors_entry = ttk.Entry(param_frame, textvariable=self.num_neighbors_var, width=5)
        num_neighbors_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Step size with validation for both comma and period
        ttk.Label(param_frame, text="Step Size:").grid(row=0, column=4, padx=5, pady=5)
        self.step_size_var = tk.StringVar(value=str(self.step_size))
        step_size_entry = ttk.Entry(param_frame, textvariable=self.step_size_var, width=5)
        step_size_entry.grid(row=0, column=5, padx=5, pady=5)
        
        # Animation speed
        ttk.Label(param_frame, text="Animation Speed:").grid(row=0, column=6, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=1.0)  # Start at 1.0 (normal speed)
        speed_scale = ttk.Scale(param_frame, from_=0.1, to=25.0, orient=tk.HORIZONTAL, 
                             length=100, variable=self.speed_var, command=self.update_speed)
        speed_scale.grid(row=0, column=7, padx=5, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        # Control buttons with modern styling - using standard tk buttons for better color control
        button_style = {"font": ("Arial", 10), "width": 10, "relief": tk.RAISED, 
                         "borderwidth": 2, "padx": 5, "pady": 5}
        
        # Use standard tk.Button instead of ttk.Button for better color control
        self.start_button = tk.Button(button_frame, text="Start", bg="#4CAF50", fg="white", 
                                     command=self.start_animation, **button_style)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = tk.Button(button_frame, text="Pause", bg="#FF9800", fg="white", 
                                     command=self.pause_animation, state=tk.DISABLED, **button_style)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.resume_button = tk.Button(button_frame, text="Resume", bg="#2196F3", fg="white", 
                                      command=self.resume_animation, state=tk.DISABLED, **button_style)
        self.resume_button.grid(row=0, column=2, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="Reset", bg="#f44336", fg="white", 
                                     command=self.reset_animation, **button_style)
        self.reset_button.grid(row=0, column=3, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(control_frame, relief=tk.GROOVE, borderwidth=1)
        status_frame.pack(pady=5, fill=tk.X, padx=20)
        
        # Status labels
        self.iteration_label = ttk.Label(status_frame, text=f"Iteration: 0 / {self.max_iterations}", 
                                      font=("Arial", 10, "bold"))
        self.iteration_label.pack(side=tk.LEFT, padx=10)
        
        self.position_label = ttk.Label(status_frame, text=f"Current Position: {self.current_x:.2f}")
        self.position_label.pack(side=tk.LEFT, padx=10)
        
        self.value_label = ttk.Label(status_frame, text=f"Current Value: {self.current_objective_function(self.current_x):.4f}")
        self.value_label.pack(side=tk.LEFT, padx=10)
    
    def change_objective_function(self, event=None):
        # Get the current selection
        selection = self.function_var.get()
        
        # Update the current objective function
        self.current_objective_function = self.objective_functions[selection]
        
        # Stop any running animation
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        # We need to reset the animation when the function changes
        try:
            self.reset_animation()
        except Exception as e:
            print(f"Error during reset: {e}")
            # Fallback if reset fails
            self.current_iteration = 0
            self.current_x = random.uniform(-6, 6)
            self.history_x = [self.current_x]
            self.history_y = [self.current_objective_function(self.current_x)]
            self.animation_stage = self.STAGE_GENERATE_NEIGHBORS
            self.algorithm_finished = False
        
        # Update the plot with the new function
        self.update_function_plot()
    
    def create_plot(self):
        # Plot frame using ttk
        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure and canvas with explicit background color
        self.fig = plt.figure(figsize=(10, 6), facecolor='white')
        self.ax = self.fig.add_subplot(111, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set initial x-axis range
        self.x_min, self.x_max = -6, 6
        
        # Initialize plot elements that will be referenced later
        current_y = self.current_objective_function(self.current_x)
        
        # Initialize current point and path
        self.current_point, = self.ax.plot([self.current_x], [current_y], 
                                        'D', color='blue', markersize=15, 
                                        markeredgecolor='black', markeredgewidth=2,
                                        label='Current Position')
        
        # Make the path more visible with brighter color, increased thickness and markers
        self.path_line, = self.ax.plot(self.history_x, self.history_y, 
                                     '-ro',  # Red line with circle markers
                                     linewidth=2.5,  # Thicker line
                                     markersize=5,   # Visible markers
                                     label='Path')
        
        # Initialize placeholder for other plot elements
        self.neighbor_points = []
        self.neighbor_lines = []
        self.neighbor_annotations = []
        self.best_neighbor_point = None
        self.best_neighbor_line = None
        self.current_annotation = None
        self.optimum_annotation = None
        
        # Plot the objective function
        self.update_objective_function_plot()
        
        # Set plot properties
        self.ax.set_title('Hill Climbing Visualization', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Position', fontsize=12)
        self.ax.set_ylabel('Objective Function Value', fontsize=12)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
    
    def update_objective_function_plot(self, update_limits=False):
        """Plot or update the objective function"""
        # Update x range if needed
        if update_limits:
            current_x = self.current_x
            # Expand view if current_x is near the edge or outside current view
            if current_x < self.x_min + 2:
                self.x_min = current_x - 4
            if current_x > self.x_max - 2:
                self.x_max = current_x + 4
        
        # Generate x values with current range
        x = np.linspace(self.x_min, self.x_max, 1000)
        y = [self.current_objective_function(xi) for xi in x]
        
        # Check if we already have a function plot line
        if hasattr(self, 'function_line') and self.function_line in self.ax.lines:
            # Update existing line
            self.function_line.set_data(x, y)
        else:
            # Create new line
            self.function_line, = self.ax.plot(x, y, 'b-', linewidth=2, label='Objective Function')
        
        # Update x-axis limits
        self.ax.set_xlim(self.x_min, self.x_max)
        
        # Update y-axis limits with some padding
        y_min, y_max = min(y), max(y)
        y_padding = (y_max - y_min) * 0.1
        self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

    def update_function_plot(self):
        # Clear the current plot but keep background white
        self.ax.clear()
        self.ax.set_facecolor('white')
        
        # Reset the x range for new function
        self.x_min, self.x_max = -6, 6
        
        # Reset the view to show the new function
        self.update_objective_function_plot()
        
        # Update current position value with new function
        current_y = self.current_objective_function(self.current_x)
        self.history_y = [current_y]
        
        # Plot current position
        self.current_point, = self.ax.plot([self.current_x], [current_y], 
                                         'D', color='blue', markersize=15, 
                                         markeredgecolor='black', markeredgewidth=2,
                                         label='Current Position')
        
        # Add current position annotation
        try:
            self.current_annotation = self.ax.annotate("CURRENT", 
                                                     xy=(self.current_x, current_y),
                                                     xytext=(0, 15), textcoords='offset points',
                                                     ha='center', va='bottom',
                                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        except Exception as e:
            print(f"Error creating initial annotation: {e}")
            self.current_annotation = None
        
        # Initialize neighbors plot
        self.neighbor_points = []
        self.neighbor_lines = []
        self.neighbor_annotations = []
        
        # Initialize path plot - Change to more visible style
        self.path_line, = self.ax.plot(self.history_x, self.history_y, 
                                     '-ro',  # Red line with circle markers
                                     linewidth=2.5,  # Thicker line
                                     markersize=5,   # Visible markers
                                     label='Path')
        
        # Set plot properties
        self.ax.set_title('Hill Climbing Visualization', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Position', fontsize=12)
        self.ax.set_ylabel('Objective Function Value', fontsize=12)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()
        
        # Update value label
        self.value_label.config(text=f"Current Value: {current_y:.4f}")
        
        # Redraw
        self.canvas.draw()
    
    def safe_remove_annotation(self, annotation):
        """Safely remove an annotation from the plot"""
        if annotation is None:
            return
        
        try:
            # Check if annotation is still in the axes
            if annotation in self.ax.texts:
                annotation.remove()
            else:
                # If not in texts list, try direct removal (might fail silently)
                try:
                    annotation.remove()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error removing annotation: {e}")
    
    def safe_remove_patch(self, patch):
        """Safely remove a patch (like axvspan) from the plot"""
        if patch is None:
            return
        
        try:
            # Check if patch is still in the axes
            if patch in self.ax.patches:
                patch.remove()
            else:
                # If not in patches list, try direct removal (might fail silently)
                try:
                    patch.remove()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error removing patch: {e}")
    
    def stage_generate_neighbors(self):
        # Clear previous neighbors
        for point in self.neighbor_points:
            point.remove() if point in self.ax.lines else None
        # Clear previous connection lines
        for line in self.neighbor_lines:
            line.remove() if line in self.ax.lines else None
        
        # Remove previous annotation safely
        self.safe_remove_annotation(self.current_annotation)
        
        self.neighbor_points = []
        self.neighbor_lines = []
        
        # Get current position value
        current_x = self.current_x
        current_y = self.current_objective_function(current_x)
        
        # Add current position annotation
        try:
            self.current_annotation = self.ax.annotate("CURRENT", 
                                                     xy=(current_x, current_y),
                                                     xytext=(0, 15), textcoords='offset points',
                                                     ha='center', va='bottom',
                                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        except Exception as e:
            print(f"Error creating annotation: {e}")
            self.current_annotation = None
        
        # Generate neighbors
        self.neighbors = []
        self.neighbor_values = []
        self.neighbor_annotations = []  # Track annotations for cleanup
        
        # Add a colored background for neighbors area
        min_neighbor = current_x - self.step_size
        max_neighbor = current_x + self.step_size
        try:
            self.neighbor_area = self.ax.axvspan(min_neighbor, max_neighbor, alpha=0.1, color='gray', zorder=0)
        except Exception as e:
            print(f"Error creating neighbor area: {e}")
            self.neighbor_area = None
        
        for i in range(self.num_neighbors):
            # Generate random neighbor within step size
            neighbor = self.current_x + (random.random() * 2 - 1) * self.step_size
            self.neighbors.append(neighbor)
            neighbor_value = self.current_objective_function(neighbor)
            self.neighbor_values.append(neighbor_value)
            
            # Display neighbors with random colors
            color = np.random.rand(3,)
            
            # Draw line connecting neighbor to current point
            line, = self.ax.plot([current_x, neighbor], [current_y, neighbor_value], 
                                color=color, linestyle='-', linewidth=1, alpha=0.5)
            self.neighbor_lines.append(line)
            
            # Display neighbor point - smaller, more subtle
            point, = self.ax.plot([neighbor], [neighbor_value], 'o', color=color, markersize=6, 
                                alpha=0.7)
            self.neighbor_points.append(point)
            
            # Add small annotation for neighbor
            annotation = self.ax.annotate(f"N{i+1}", 
                                       xy=(neighbor, neighbor_value),
                                       xytext=(0, -10), textcoords='offset points',
                                       ha='center', va='top', fontsize=8)
            self.neighbor_annotations.append(annotation)
        
        # Find best neighbor
        best_idx = np.argmax(self.neighbor_values)
        self.best_neighbor = self.neighbors[best_idx]
        self.best_neighbor_value = self.neighbor_values[best_idx]
        
        # Store points to remove in the next stage (all except best)
        self.neighbors_to_remove = []
        for i in range(len(self.neighbors)):
            if i != best_idx:
                self.neighbors_to_remove.append((self.neighbor_points[i], self.neighbor_lines[i]))
        
        # Store best neighbor point and line for next stage
        self.best_neighbor_point = self.neighbor_points[best_idx]
        self.best_neighbor_line = self.neighbor_lines[best_idx]
        
        # Move to next stage
        self.animation_stage = self.STAGE_FIND_BEST
        
        # Refresh canvas
        self.canvas.draw()
        
        return self.current_point, self.path_line
    
    def stage_find_best(self):
        # Remove one neighbor at a time for animation effect
        if self.neighbors_to_remove:
            point, line = self.neighbors_to_remove.pop()
            point.remove() if point in self.ax.lines else None
            line.remove() if line in self.ax.lines else None
            
            # If we've removed all but the best, prepare for transition
            if not self.neighbors_to_remove:
                # Check if the best neighbor is actually better than current position
                current_value = self.current_objective_function(self.current_x)
                
                if self.best_neighbor_value > current_value:
                    # Only highlight best neighbor if it's better than current position
                    # Change color of best neighbor to highlight it
                    self.best_neighbor_point.set_color('yellow')
                    self.best_neighbor_point.set_markersize(10)
                    self.best_neighbor_point.set_markeredgecolor('black')
                    self.best_neighbor_point.set_markeredgewidth(1.5)
                    
                    # Change color of the line to best neighbor
                    self.best_neighbor_line.set_color('yellow')
                    self.best_neighbor_line.set_alpha(0.8)
                    self.best_neighbor_line.set_linewidth(2)
                else:
                    # If no better neighbor exists, don't highlight it
                    # Just make it slightly more visible than other neighbors
                    self.best_neighbor_point.set_color('orange')
                    self.best_neighbor_point.set_markersize(8)
                    self.best_neighbor_point.set_alpha(0.6)
                    
                    # Make the connection line less prominent
                    self.best_neighbor_line.set_color('gray')
                    self.best_neighbor_line.set_alpha(0.3)
                    self.best_neighbor_line.set_linewidth(1)
                
                # Set up transition
                self.transition_current_step = 0
                self.animation_stage = self.STAGE_TRANSITION
        
        # Refresh canvas
        self.canvas.draw()
        
        return self.current_point, self.path_line
    
    def stage_transition(self):
        # Animate transition from current to best (if best is better)
        current_value = self.current_objective_function(self.current_x)
        
        # Only transition if best is better than current
        if self.best_neighbor_value > current_value:
            # Calculate step size for transition
            self.transition_current_step += 1
            progress = self.transition_current_step / self.transition_steps
            
            # Interpolate between current and best positions
            interp_x = self.current_x + (self.best_neighbor - self.current_x) * progress
            interp_y = self.current_objective_function(interp_x)
            
            # Update current point position
            self.current_point.set_data([interp_x], [interp_y])
            
            # Update annotation position
            self.safe_remove_annotation(self.current_annotation)
            try:
                self.current_annotation = self.ax.annotate("CURRENT", 
                                                         xy=(interp_x, interp_y),
                                                         xytext=(0, 15), textcoords='offset points',
                                                         ha='center', va='bottom',
                                                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                                                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
            except Exception as e:
                print(f"Error creating transition annotation: {e}")
                self.current_annotation = None
            
            # If transition is complete, update path and prepare for next iteration
            if self.transition_current_step >= self.transition_steps:
                # Remove best neighbor point and line
                self.best_neighbor_point.remove() if self.best_neighbor_point in self.ax.lines else None
                self.best_neighbor_line.remove() if self.best_neighbor_line in self.ax.lines else None
                
                # Remove neighbor area background
                self.safe_remove_patch(self.neighbor_area)
                
                # Remove neighbor annotations
                for ann in self.neighbor_annotations:
                    ann.remove()
                self.neighbor_annotations = []
                
                # Update current position
                self.current_x = self.best_neighbor
                self.history_x.append(self.current_x)
                self.history_y.append(self.best_neighbor_value)
                
                # Update path
                self.path_line.set_data(self.history_x, self.history_y)
                
                # Update status labels and iteration counter
                self.current_iteration += 1
                self.iteration_label.config(text=f"Iteration: {self.current_iteration} / {self.max_iterations}")
                self.position_label.config(text=f"Current Position: {self.current_x:.2f}")
                self.value_label.config(text=f"Current Value: {self.current_objective_function(self.current_x):.4f}")
                
                # Reset for next iteration
                self.animation_stage = self.STAGE_GENERATE_NEIGHBORS
                
                # Check if we need to expand the plot view
                if self.current_x < self.x_min + 1 or self.current_x > self.x_max - 1:
                    # Current position is getting close to the edge, expand the view
                    self.update_objective_function_plot(update_limits=True)
        else:
            # If best not better than current, skip transition and stop algorithm
            # Remove best neighbor point and line
            self.best_neighbor_point.remove() if self.best_neighbor_point in self.ax.lines else None
            self.best_neighbor_line.remove() if self.best_neighbor_line in self.ax.lines else None
            
            # Remove neighbor area background
            self.safe_remove_patch(self.neighbor_area)
            
            # Remove neighbor annotations
            for ann in self.neighbor_annotations:
                ann.remove()
            self.neighbor_annotations = []
            
            # Update status labels and mark algorithm as finished
            self.current_iteration += 1
            self.stop_reason = "FOUND_OPTIMUM"
            self.iteration_label.config(text=f"Iteration: {self.current_iteration} / {self.max_iterations} - STOPPED (Local optimum found)")
            self.position_label.config(text=f"Current Position: {self.current_x:.2f}")
            self.value_label.config(text=f"Current Value: {self.current_objective_function(self.current_x):.4f}")
            
            # Add indicator for local optimum
            try:
                self.optimum_annotation = self.ax.annotate("FOUND OPTIMUM", 
                                                        xy=(self.current_x, self.current_objective_function(self.current_x)),
                                                        xytext=(0, 30), textcoords='offset points',
                                                        ha='center', va='bottom', fontsize=12,
                                                        bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="black", alpha=0.8),
                                                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
            except Exception as e:
                print(f"Error creating optimum annotation: {e}")
                self.optimum_annotation = None
            
            # Mark algorithm as finished
            self.algorithm_finished = True
            
            # Reset for next iteration (won't be used due to algorithm_finished flag)
            self.animation_stage = self.STAGE_GENERATE_NEIGHBORS
        
        # Refresh canvas
        self.canvas.draw()
        
        return self.current_point, self.path_line
    
    def validate_inputs(self):
        """Validate all input values and return True if all are valid, False otherwise."""
        self.validation_errors = []
        
        # Validate max iterations (positive integer)
        try:
            max_iter = self.max_iter_var.get()
            if not isinstance(max_iter, int) or max_iter <= 0:
                self.validation_errors.append("Maximum iterations must be a positive integer.")
        except (ValueError, tk.TclError):
            self.validation_errors.append("Maximum iterations must be a valid number.")
            
        # Validate number of neighbors (positive integer)
        try:
            num_neighbors = self.num_neighbors_var.get()
            if not isinstance(num_neighbors, int) or num_neighbors <= 0:
                self.validation_errors.append("Number of neighbors must be a positive integer.")
        except (ValueError, tk.TclError):
            self.validation_errors.append("Number of neighbors must be a valid number.")
        
        # Validate step size (positive float)
        try:
            step_size_str = self.step_size_var.get().replace(',', '.')
            step_size = float(step_size_str)
            if step_size <= 0:
                self.validation_errors.append("Step size must be a positive number.")
        except (ValueError, tk.TclError):
            self.validation_errors.append("Step size must be a valid number.")
        
        # Validate animation speed (positive float)
        try:
            anim_speed = self.speed_var.get()
            if anim_speed <= 0:
                self.validation_errors.append("Animation speed must be a positive number.")
        except (ValueError, tk.TclError):
            self.validation_errors.append("Animation speed must be a valid number.")
        
        # If there are errors, show them and return False
        if self.validation_errors:
            error_message = "Please correct the following errors:\n• " + "\n• ".join(self.validation_errors)
            messagebox.showerror("Input Validation Error", error_message)
            return False
        
        return True
    
    def start_animation(self):
        # Validate inputs before starting
        if not self.validate_inputs():
            return
            
        # Apply updated parameters
        self.max_iterations = self.max_iter_var.get()
        self.num_neighbors = self.num_neighbors_var.get()
        
        # Get step size with support for comma or period
        self.step_size = self.get_float_value(self.step_size_var.get())
        # Update the entry with the standardized format (using period)
        self.step_size_var.set(str(self.step_size))
        
        # Reset iteration label to show max iterations
        self.iteration_label.config(text=f"Iteration: {self.current_iteration} / {self.max_iterations}")
        
        # Reset if needed
        if self.current_iteration >= self.max_iterations:
            self.reset_animation()
        
        self.animation_paused = False
        
        # Start animation if not already running
        if self.animation is None:
            # Calculate interval based on animation speed
            if self.anim_speed <= 1.0:
                interval = int(self.base_interval / self.anim_speed)
                interval = min(interval, 2000)
            else:
                interval = int(self.base_interval / (self.anim_speed * self.anim_speed))
                interval = max(interval, 10)
            
            # Add save_count parameter to avoid unbounded caching warning
            self.animation = FuncAnimation(self.fig, self.update_plot, interval=interval, 
                                          blit=False, repeat=False, save_count=self.max_iterations * 3)
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
        
        # Start animation
        self.canvas.draw()
    
    def pause_animation(self):
        self.animation_paused = True
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
    
    def resume_animation(self):
        # Validate inputs before resuming
        if not self.validate_inputs():
            return
            
        self.animation_paused = False
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
    
    def reset_animation(self):
        # Stop animation
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        
        # Reset algorithm state
        self.current_iteration = 0
        self.current_x = random.uniform(-6, 6)
        self.history_x = [self.current_x]
        self.history_y = [self.current_objective_function(self.current_x)]
        self.animation_stage = self.STAGE_GENERATE_NEIGHBORS
        self.algorithm_finished = False
        self.stop_reason = None
        
        # Clear neighbors and annotations
        for point in self.neighbor_points:
            point.remove() if point in self.ax.lines else None
        for line in self.neighbor_lines:
            line.remove() if line in self.ax.lines else None
        if hasattr(self, 'best_neighbor_point') and self.best_neighbor_point and self.best_neighbor_point in self.ax.lines:
            self.best_neighbor_point.remove()
        if hasattr(self, 'best_neighbor_line') and self.best_neighbor_line and self.best_neighbor_line in self.ax.lines:
            self.best_neighbor_line.remove()
        
        # Remove neighbor area background
        self.safe_remove_patch(self.neighbor_area)
        
        # Remove neighbor annotations
        if hasattr(self, 'neighbor_annotations'):
            for ann in self.neighbor_annotations:
                ann.remove()
            self.neighbor_annotations = []
        
        self.neighbor_points = []
        self.neighbor_lines = []
        
        # Remove current annotation if it exists
        self.safe_remove_annotation(self.current_annotation)
        
        # Remove local optimum annotation if it exists
        self.safe_remove_annotation(self.optimum_annotation)
        self.optimum_annotation = None
        
        # Update plots
        self.current_point.set_data([self.current_x], [self.history_y[0]])
        self.path_line.set_data(self.history_x, self.history_y)
        
        # Add new annotation
        current_y = self.current_objective_function(self.current_x)
        try:
            self.current_annotation = self.ax.annotate("CURRENT", 
                                                     xy=(self.current_x, current_y),
                                                     xytext=(0, 15), textcoords='offset points',
                                                     ha='center', va='bottom',
                                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
                                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        except Exception as e:
            print(f"Error creating reset annotation: {e}")
            self.current_annotation = None
        
        # Reset status
        self.iteration_label.config(text=f"Iteration: 0 / {self.max_iterations}")
        self.position_label.config(text=f"Current Position: {self.current_x:.2f}")
        self.value_label.config(text=f"Current Value: {self.current_objective_function(self.current_x):.4f}")
        
        # Reset buttons
        self.animation_paused = True
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        
        # Redraw
        self.canvas.draw()
        
        # Get step size with support for comma or period
        try:
            self.step_size = self.get_float_value(self.step_size_var.get())
            # Update the entry with the standardized format (using period)
            self.step_size_var.set(str(self.step_size))
        except Exception as e:
            print(f"Error parsing step size: {e}")
    
    def update_plot(self, frame):
        if self.animation_paused:
            return self.current_point, self.path_line
        
        if self.current_iteration >= self.max_iterations and self.animation_stage == self.STAGE_GENERATE_NEIGHBORS:
            # Stop due to reaching iteration limit
            self.pause_animation()
            self.stop_reason = "ITERATION_LIMIT"
            
            # Make sure we display the correct iteration count (max iterations)
            self.current_iteration = self.max_iterations
            self.iteration_label.config(text=f"Iteration: {self.current_iteration} / {self.max_iterations} - STOPPED (Iteration limit reached)")
            
            # Add indicator for reaching iteration limit
            try:
                self.optimum_annotation = self.ax.annotate("ITERATION LIMIT REACHED", 
                                                       xy=(self.current_x, self.current_objective_function(self.current_x)),
                                                       xytext=(0, 30), textcoords='offset points',
                                                       ha='center', va='bottom', fontsize=12,
                                                       bbox=dict(boxstyle="round,pad=0.3", fc="orange", ec="black", alpha=0.8),
                                                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
            except Exception as e:
                print(f"Error creating limit annotation: {e}")
                
            return self.current_point, self.path_line
        
        if self.algorithm_finished:
            self.pause_animation()
            return self.current_point, self.path_line
        
        # Handle different animation stages
        if self.animation_stage == self.STAGE_GENERATE_NEIGHBORS:
            return self.stage_generate_neighbors()
        elif self.animation_stage == self.STAGE_FIND_BEST:
            return self.stage_find_best()
        elif self.animation_stage == self.STAGE_TRANSITION:
            return self.stage_transition()
    
    def update_speed(self, value):
        """Update animation speed based on scale value"""
        # Convert scale value to speed multiplier
        self.anim_speed = float(value)
        
        # Use more extreme speed range - from 50ms to 2000ms
        # This allows for much faster animations at high speed settings
        if self.anim_speed <= 1.0:
            # Below 1.0, slow down more dramatically (500ms to 2000ms)
            interval = int(self.base_interval / self.anim_speed)
            # Cap at 2000ms to prevent extreme slowness
            interval = min(interval, 2000)
        else:
            # Above 1.0, scale down aggressively for faster speeds
            # Map 1.0 -> 500ms, 5.0 -> 50ms, 25.0 -> 10ms
            interval = int(self.base_interval / (self.anim_speed * self.anim_speed))
            # Ensure minimum of 10ms for very fast speeds
            interval = max(interval, 10)
        
        # Update animation interval if animation is running
        if self.animation:
            self.animation.event_source.interval = interval
    
    def get_float_value(self, string_value):
        """Convert a string to float, supporting both period and comma as decimal separator"""
        # Replace comma with period to handle either decimal separator
        cleaned_value = string_value.replace(',', '.')
        try:
            value = float(cleaned_value)
            if value <= 0:
                raise ValueError("Value must be positive")
            return value
        except ValueError:
            # Return default value if conversion fails
            print(f"Invalid numeric value: {string_value}, using default")
            return 0.5
    
    def exit_application(self, event=None):
        """Exit the application gracefully"""
        print("Exiting application (Ctrl+C pressed)")
        if self.animation:
            self.animation.event_source.stop()
        self.root.quit()
        self.root.destroy()

def main():
    if HAS_THEMED_TK:
        # Use themed Tk for a modern look
        root = ThemedTk(theme="arc")  # Options: "arc", "equilux", "breeze", etc.
    else:
        # Fall back to standard Tk
        root = tk.Tk()
    
    # Set up Ctrl+C handling at the top level as well
    root.bind("<Control-c>", lambda event: root.destroy())
    
    app = HillClimbingVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by keyboard interrupt (Ctrl+C)")
        # Ensure clean exit
        import sys
        sys.exit(0)