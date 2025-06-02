import plotly.graph_objects as go
import argparse
import sys
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, "../results")

# python plot.py (d)

# Argument parsing
parser = argparse.ArgumentParser(description='Plot 3D points from output files.')
parser.add_argument('--epsilon', type=float, help='Epsilon value for the problem', default=0.5)
parser.add_argument('--results_path', type=str, help='Path to the outputs directory', default=results_path)
if len(sys.argv) == 1:
    print("No arguments provided. Using default epsilon=2.0 and results_path.")

args = parser.parse_args()

epsilon = args.epsilon

# read outputP.txt and outputQ.txt

max_val = 0
min_val = 0

P = []
with open(os.path.join(results_path, "outputP.txt")) as f:
    for line in f:
        x, y, z = map(float, line.split())
        max_val = max(max_val, x+epsilon, y+epsilon, z+epsilon)
        min_val = min(min_val, x-epsilon, y-epsilon, z-epsilon)
        P.append((x, y, z))

Q = []
with open(os.path.join(results_path, "outputQ.txt")) as f:
    for line in f:
        x, y, z = map(float, line.split())
        max_val = max(max_val, x+epsilon, y+epsilon, z+epsilon)
        min_val = min(min_val, x-epsilon, y-epsilon, z-epsilon)
        Q.append((x, y, z))

print(f"P: {len(P)} points")
print(f"Q: {len(Q)} points")

def create_sphere(radius, x_center, y_center, z_center, color, text=""):
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x = x_center + radius * np.sin(phi) * np.cos(theta)
    y = y_center + radius * np.sin(phi) * np.sin(theta)
    z = z_center + radius * np.cos(phi)
    return go.Mesh3d(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        legendgroup='P', showlegend=False,
        alphahull=0, opacity=0.1, color=color, text=text
    )

# Plotting

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in P],
    y=[p[1] for p in P],
    z=[p[2] for p in P],
    mode='markers',
    name='P',
    legendgroup='P',
    text = [f"P{i+1}" for i in range(len(P))],
    marker=dict(
        size=3,
        color='red',
        opacity=0.8,
    )
))

radius = epsilon

for i in range(len(P)):
    fig.add_trace(create_sphere(radius, P[i][0], P[i][1], P[i][2], "red", f"a ball centered at P{i+1} (radius={radius})"))


for i in range(len(P)-1):
    fig.add_trace(go.Scatter3d(
        x=[P[i][0], P[i+1][0]],
        y=[P[i][1], P[i+1][1]],
        z=[P[i][2], P[i+1][2]],
        mode='lines',
        name='P',
        legendgroup='P',
        showlegend=False,
        line=dict(
            color='red',
            width=0.5
        )
    ))


fig.add_trace(go.Scatter3d(
    x=[q[0] for q in Q],
    y=[q[1] for q in Q],
    z=[q[2] for q in Q],
    mode='markers',
    name='Q',
    legendgroup='Q',
    text = [f"Q{i+1}" for i in range(len(Q))],
    marker=dict(
        size=3,
        color='blue',
        opacity=0.8,
    )
))


for i in range(len(Q)-1):
    fig.add_trace(go.Scatter3d(
        x=[Q[i][0], Q[i+1][0]],
        y=[Q[i][1], Q[i+1][1]],
        z=[Q[i][2], Q[i+1][2]],
        mode='lines',
        name='Q',
        legendgroup='Q',
        showlegend=False,
        line=dict(
            color='blue',
            width=1
        )
    ))

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val]),
        zaxis=dict(range=[min_val, max_val]),
        aspectmode='cube',
    )
)

# output to HTML
fig.write_html("index.html")

print("Plot saved to index.html")
