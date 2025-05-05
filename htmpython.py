import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
import os



# Define folder names
FOLDERS = ["dx", "dy", "dz", "thetax", "thetay", "thetaz"]

def find_csv_files(base_dir):
    """
    Searches for CSV files in the script's directory.

    Parameters:
        base_dir (str): directory where folders are located

    Returns:
        dict: A dictionary where keys are folder names and values are CSV file paths
    """
    csv_files = {}

    for folder in FOLDERS:
        folder_path = os.path.join(base_dir, folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))

        if files:
            csv_files[folder] = files[0]  # Take the first CSV file found

    return csv_files

def csv_to_numpy(file_path):
    """
    Reads a CSV file and converts it into a NumPy array

    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        numpy.ndarray: NumPy array of the CSV data
    """
    df = pd.read_csv(file_path)
    return df.to_numpy()

def htm(theta_x, theta_y, theta_z, dx, dy, dz):
    """4x4 htm."""
    cx, cy, cz = np.cos(np.radians([theta_x, theta_y, theta_z]))
    sx, sy, sz = np.sin(np.radians([theta_x, theta_y, theta_z]))

    T = np.array([
        [cy * cz, cz * sx * sy - cx * sz, cz * cx * sy + sx * sz, dx],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx, dy],
        [-sy,     cy * sx,                cy * cx,               dz],
        [0,       0,                      0,                     1]
    ])
    return T

def addxtotranspoints(transpoints, x, i):
    transpoints[0]+=x[i]
    return transpoints

def plotmovement(x, dx, dy, dz, thetax, thetay, thetaz, points, name):

    # Store transformed positions
    transformed_points_1 = []
    transformed_points_2 = []
    transformed_points_3 = []
    transformed_points_4 = []
    transformed_points_origin = []

    regpoints1 = []
    regpoints2 = []
    regpoints3 = []
    regpoints4 = []
    regpointsorigin = []

    for i in range(len(x)):
        # Compute transformation matrix for current step

        T = htm(thetax[i], thetay[i], thetaz[i], dx[i], dy[i], dz[i])

        #print(T)

        # Apply transformation to all points
        transformed_1 = T @ points[0]  # Transform first point
        transformed_2 = T @ points[1]  # Transform second point
        transformed_3 = T @ points[2] # Transform third point
        transformed_4 = T @ points[3] # Transform fourth point
        transformed_origin = T @ points[4] # Transform origin point
        
        transformed_1 = addxtotranspoints(transformed_1, x, i)
        transformed_2 = addxtotranspoints(transformed_2, x, i)
        transformed_3 = addxtotranspoints(transformed_3, x, i) 
        transformed_4 = addxtotranspoints(transformed_4, x, i)

        
        # Store transformed coordinates (exclude homogeneous 1)
        transformed_points_1.append(transformed_1[:3])
        transformed_points_2.append(transformed_2[:3])
        transformed_points_3.append(transformed_3[:3])
        transformed_points_4.append(transformed_4[:3])
        transformed_points_origin.append(transformed_origin[:3])

        #store non-transformed points
        regpoints1.append(points[0][:3])
        regpoints2.append(points[1][:3])
        regpoints3.append(points[2][:3])
        regpoints4.append(points[3][:3])
        regpointsorigin.append(points[4][:3])

    # Convert lists to NumPy arrays
    transformed_points_1 = np.array(transformed_points_1)
    transformed_points_2 = np.array(transformed_points_2)
    transformed_points_3 = np.array(transformed_points_3)
    transformed_points_4 = np.array(transformed_points_4)
    transformed_points_origin = np.array(transformed_points_origin)

    print(transformed_points_1)
    print(transformed_points_2)
    print(transformed_points_3)
    print(transformed_points_4)
    print(transformed_points_origin)

    regpoints1 = np.array(regpoints1)
    regpoints2 = np.array(regpoints2)
    regpoints3 = np.array(regpoints3)
    regpoints4 = np.array(regpoints4)
    regpointsorigin = np.array(regpointsorigin)


    startxs = []
    startys = []
    startzs = []
    endxs = []
    realstartxs = []
    realstartys = []
    realstartzs = []
    realendxs = []
    realendys = []
    realendzs = []

    #create start and end rectangles----------------------------------------
    for _ in range(len(points)-1):
        startxs.append(points[_][0])
        startys.append(points[_][1])
        startzs.append(points[_][2])
        endxs.append(points[_][0]+x[-1])

    #add a start point again to close the loop
    startxs.append(points[0][0])
    startys.append(points[0][1])
    startzs.append(points[0][2])
    endxs.append(points[0][0]+x[-1])

    #these are rhe end points of the carriage movement to make the rectangle
    realendxs.append(transformed_points_1[-1][0])
    realendys.append(transformed_points_1[-1][1])
    realendzs.append(transformed_points_1[-1][2])

    realendxs.append(transformed_points_2[-1][0])
    realendys.append(transformed_points_2[-1][1])
    realendzs.append(transformed_points_2[-1][2])

    realendxs.append(transformed_points_3[-1][0])
    realendys.append(transformed_points_3[-1][1])
    realendzs.append(transformed_points_3[-1][2])

    realendxs.append(transformed_points_4[-1][0])
    realendys.append(transformed_points_4[-1][1])
    realendzs.append(transformed_points_4[-1][2])

    realendxs.append(transformed_points_1[-1][0])
    realendys.append(transformed_points_1[-1][1])
    realendzs.append(transformed_points_1[-1][2])

    #starting rectangle real
    realstartxs.append(transformed_points_1[0][0])
    realstartys.append(transformed_points_1[0][1])
    realstartzs.append(transformed_points_1[0][2])

    realstartxs.append(transformed_points_2[0][0])
    realstartys.append(transformed_points_2[0][1])
    realstartzs.append(transformed_points_2[0][2])

    realstartxs.append(transformed_points_3[0][0])
    realstartys.append(transformed_points_3[0][1])
    realstartzs.append(transformed_points_3[0][2])

    realstartxs.append(transformed_points_4[0][0])
    realstartys.append(transformed_points_4[0][1])
    realstartzs.append(transformed_points_4[0][2])

    realstartxs.append(transformed_points_1[0][0])
    realstartys.append(transformed_points_1[0][1])
    realstartzs.append(transformed_points_1[0][2])
    #------------------------------------------------------------------

    # Plotting transformed points
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot transformed paths
    ax.plot(transformed_points_1[:, 0], transformed_points_1[:, 1], transformed_points_1[:, 2], 'r-', label="Path of points")
    ax.plot(transformed_points_2[:, 0], transformed_points_2[:, 1], transformed_points_2[:, 2], 'r-')
    ax.plot(transformed_points_3[:, 0], transformed_points_3[:, 1], transformed_points_3[:, 2], 'r-')
    ax.plot(transformed_points_4[:, 0], transformed_points_4[:, 1], transformed_points_4[:, 2], 'r-')
    #ax.plot(transformed_points_origin[:, 0], transformed_points_origin[:, 1], transformed_points_origin[:, 2], 'r-')
    # Plot ideal paths
    print(f"x - {x}")
    print(regpoints1[:, 1])
    ax.plot(x, regpoints1[:, 1], regpoints1[:, 2], 'g-', label="Path of nominal points")
    ax.plot(x, regpoints2[:, 1], regpoints2[:, 2], 'g-')
    ax.plot(x, regpoints3[:, 1], regpoints3[:, 2], 'g-')
    ax.plot(x, regpoints4[:, 1], regpoints4[:, 2], 'g-')
    #ax.plot(dx, regpointsorigin[:, 1], regpointsorigin[:, 2], 'g-')

    # Plot start and end points(ideal)
    ax.plot(startxs, startys, startzs, 'g-')
    ax.plot(endxs, startys, startzs, 'g-')

    ax.plot(realstartxs, realstartys, realstartzs, 'r-')
    ax.plot(realendxs, realendys, realendzs, 'r-')

    # Labels and title
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"{name} 3D Transformation Paths")


    # Show legend
    ax.legend()

    # Show plot
    plt.show()

def findaccuracy(dx, dy, ploton, name):
    # Step 1: Find the slope using least squares fit (initial estimate)
    coeffs = np.polyfit(dx, dy, 1)  # Linear least squares fit
    a_est = coeffs[0]

    reference_line = np.polyval(coeffs, dx)


    # Step 2: Compute intercepts for the minimum zone lines
    # The upper and lower lines are parallel with different intercepts
    b_upper = np.max(dy - a_est * dx)  # Max deviation above the estimated line
    b_lower = np.min(dy - a_est * dx)  # Min deviation below the estimated line

    truestraightness = dy-reference_line

    # Compute the two boundary lines
    y_upper = a_est * dx + b_upper
    y_lower = a_est * dx + b_lower

    # Compute the minimum straightness error as the difference between the bounds
    min_straightness_error = b_upper - b_lower

    df= pd.DataFrame({"X":dx, "Y": dy})
    avg_df = df.groupby("X", as_index=False).mean()

    df2= pd.DataFrame({"X":dx, "Y": truestraightness})
    avg_df2 = df2.groupby("X", as_index=False).mean()

    if ploton:
        # Plot the deviation data and minimum zone fit
        plt.figure(figsize=(8, 5))

        plt.scatter(df["X"], df["Y"], c='b', label=f"Raw Data")
        plt.plot(avg_df["X"], avg_df["Y"], 'b-', label="Measurement Average")

        plt.plot(dx, reference_line, linestyle='-', color='red', label="Reference Straight Line")
        plt.plot(dx, y_upper, '--g', label="Minimum Zone Bounds")
        plt.plot(dx, y_lower, '--g')

        # Labels and legend
        plt.xlabel("Nominal Travel Position (dx)")
        plt.ylabel("Straightness Deviation")
        plt.title(f"{name} Straightness Deviation")
        plt.axhline(0, color='black', linewidth=0.5)  # Horizontal reference
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()

        # Plot the true straightness error
        plt.figure(figsize=(8, 5))
        plt.plot(avg_df2["X"], avg_df2["Y"], 'r-', label="True Straightness Error")

        # Labels and legend
        plt.xlabel("Nominal Travel Position (dx)")
        plt.ylabel("True Straightness Error")
        plt.title(f"{name} Straightness Deviation")
        plt.axhline(0, color='black', linewidth=2)  # Horizontal reference
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()

    # Output the minimum straightness error
    return np.array(avg_df2["X"]), np.array(avg_df2["Y"]), min_straightness_error, a_est


def findstraightness(dx, dy, ploton, name):
    # Step 1: Find the slope using least squares fit (initial estimate)
    coeffs = np.polyfit(dx, dy, 1)  # Linear least squares fit
    a_est = coeffs[0]

    reference_line = np.polyval(coeffs, dx)


    # Step 2: Compute intercepts for the minimum zone lines
    # The upper and lower lines are parallel with different intercepts
    b_upper = np.max(dy - a_est * dx)  # Max deviation above the estimated line
    b_lower = np.min(dy - a_est * dx)  # Min deviation below the estimated line

    truestraightness = dy-reference_line

    # Compute the two boundary lines
    y_upper = a_est * dx + b_upper
    y_lower = a_est * dx + b_lower

    # Compute the minimum straightness error as the difference between the bounds
    min_straightness_error = b_upper - b_lower

    df= pd.DataFrame({"X":dx, "Y": dy})
    avg_df = df.groupby("X", as_index=False).mean()

    df2= pd.DataFrame({"X":dx, "Y": truestraightness})
    avg_df2 = df2.groupby("X", as_index=False).mean()

    if ploton:
        # Plot the deviation data and minimum zone fit
        plt.figure(figsize=(8, 5))

        plt.scatter(df["X"], df["Y"], c='b', label=f"Raw Data")
        plt.plot(avg_df["X"], avg_df["Y"], 'b-', label="Measurement Average")

        plt.plot(dx, reference_line, linestyle='-', color='red', label="Reference Straight Line")
        plt.plot(dx, y_upper, '--g', label="Minimum Zone Bounds")
        plt.plot(dx, y_lower, '--g')

        # Labels and legend
        plt.xlabel("Nominal Travel Position (dx)")
        plt.ylabel("Straightness Deviation")
        plt.title(f"{name} Straightness Deviation")
        plt.axhline(0, color='black', linewidth=0.5)  # Horizontal reference
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()

        # Plot the true straightness error
        plt.figure(figsize=(8, 5))
        plt.plot(avg_df2["X"], avg_df2["Y"], 'r-', label="True Straightness Error")

        # Labels and legend
        plt.xlabel("Nominal Travel Position (dx)")
        plt.ylabel("True Straightness Error")
        plt.title(f"{name} Straightness Deviation")
        plt.axhline(0, color='black', linewidth=2)  # Horizontal reference
        plt.legend()
        plt.grid()

        # Display the plot
        plt.show()

    # Output the minimum straightness error
    return np.array(avg_df2["X"]), np.array(avg_df2["Y"]), min_straightness_error, a_est

def findangularerror(dx, theta, ploton, name, units):
        
    max_roll_error = np.max(np.abs(theta))

    df= pd.DataFrame({"X":dx, "Y": theta})
    avg_df = df.groupby("X", as_index=False).mean()

    if ploton:
            # Create plots for angular errors
        plt.figure(figsize=(10, 6))

        # Roll (θx) Error Plot
        plt.scatter(df["X"], df["Y"], c='r', label=f"θ{name}")
        plt.plot(avg_df["X"], avg_df["Y"], 'r-', label="Average")

        # Labels and legend
        plt.xlabel("Nominal Travel Position (dx)")
        plt.ylabel(f"Angular Error ({units})")
        plt.title(f"Angular Error Around {name} Axis")
        plt.axhline(0, color='black', linewidth=2)  # Horizontal reference
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()

    return np.array(avg_df["X"]), np.array(avg_df["Y"]), max_roll_error

def interpolate_to_match(x_original, y_original, x_target):
    """
    Interpolates y-values to match a new set of x-values.

    Parameters:
        x_original (array-like): Original x-values.
        y_original (array-like): Original y-values corresponding to x_original.
        x_target (array-like): The x-values to interpolate y-values onto.

    Returns:
        np.ndarray: Interpolated y-values matching x_target.
    """
    return np.interp(x_target, x_original, y_original)

def pullvalsfromarray(x, np_arrays, name, scale):
    zdz = []
    zdx = []
    if name in np_arrays:
        for run in np_arrays[name]:
            zdx.append(run[2]*scale) #3rd column, with target value
            zdz.append(run[3]) #4th column, with error value
        zdz=np.array(zdz)
        zdx=np.array(zdx)
    else:
        zdz = np.array([0]*(len(x)))
        zdx = x
    return zdx, zdz

def getx(np_arrays, name, scale):
    x = []
    xdx = []
    for run in np_arrays[name]:
        x.append(run[2]*scale) #3rd column, with target value
        xdx.append(run[3]) #4th column, with error value
    x=np.array(x)
    xdx=np.array(xdx)

    return x, xdx

        
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script directory
    csv_files = find_csv_files(script_dir)

    ploton = True
    
    if csv_files:
        np_arrays = {}

        for folder, file_path in csv_files.items():
            #   print(f"Found CSV file in '{folder}': {file_path}")
            np_arrays[folder] = csv_to_numpy(file_path)

        # print("\nConverted NumPy arrays:")
        #   for folder, np_array in np_arrays.items():
            #   print(f"\nFolder: {folder}\n{np_array}")

        
    else:
        print("No CSV files found in any of the specified folders.")


    xspace = np.linspace(0, 100)

    #box dimensions in microns
    carriagex = 400
    carriagey = 800
    carriagez = 100

    #starting coordinates for box... final value is the midpoint 
    boxcorners = np.array([
        [0, -carriagey/2, carriagez/2, 1], 
        [0, carriagey/2, carriagez/2, 1], 
        [0, carriagey/2, -carriagez/2, 1],
        [0, -carriagey/2, -carriagez/2, 1],
        [0, 0, 0, 1]

    ])

    scale = 1000
    descale = 0.001

    x, xdx = getx(np_arrays, "dz", scale)
    xdy, ydy = pullvalsfromarray(x, np_arrays, "dy", scale)
    xdz, zdz = pullvalsfromarray(x, np_arrays, "dx", scale)
    xtx, ttx = pullvalsfromarray(x, np_arrays, "thetaz", scale) 
    xty, tty = pullvalsfromarray(x, np_arrays, "thetay", scale) 
    xtz, ttz = pullvalsfromarray(x, np_arrays, "thetax", scale) 

    
    xavg, xdxavg, xstraightness, sx = findstraightness(x, xdx, ploton, 'Z')
    xdyavg, ydyavg, ystraightness, sy = findstraightness(xdy, ydy, ploton, 'Y')
    xdzavg, zdzavg, zstraightness, sz = findstraightness(xdz, zdz, ploton, 'X')
    xtxavg, ttxavg, zangular = findangularerror(xtx, ttx, ploton, 'Z', 'degrees')
    xtyavg, ttyavg, zangular = findangularerror(xty, tty, ploton, 'Y', 'degrees')
    xtzavg, ttzavg, zangular = findangularerror(xtz, ttz, ploton, 'X', 'degrees')

    for x in range(len(xavg)):
        xavg[x] = xavg[x]*descale
        xdxavg[x] = xdxavg[x]*descale
        xdyavg[x] = xdyavg[x]*descale
        xdzavg[x] = xdzavg[x]*descale
        xtxavg[x] = xtxavg[x]*descale
        xtyavg[x] = xtyavg[x]*descale
        xtzavg[x] = xtzavg[x]*descale
    

    plotmovement(xavg, xdxavg, ydyavg, zdzavg, ttxavg, ttyavg, ttzavg, boxcorners, "True")