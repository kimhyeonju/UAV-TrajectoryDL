import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('/Users/admin/PycharmProjects/GP_test/data/split_10s/Flight Wolfie Dubai to LAX 54195_split_10s.csv')

# Extract necessary columns
latitude = df['Latitude']
longitude = df['Longitude']
altitude = df['Altitude (m)']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the data
ax.plot(latitude, longitude, altitude, color='blue', linewidth=0.5)

# Set labels
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Altitude (m)')

# Show plot
plt.show()
