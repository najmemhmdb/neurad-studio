# DataParser Visualization Tool

A standalone visualization utility for checking if poses of cameras, lidars, and point clouds are correctly loaded by the `SimulatorDataParser`.

## Features

- **Camera Frustums**: Visualizes camera poses with clickable frustums and image thumbnails
- **Image Display**: Shows actual camera images in the frustums for easy verification
- **LiDAR Point Clouds**: Displays lidar point clouds with intensity-based colors
- **LiDAR Poses**: Shows coordinate frames for each lidar scan
- **Actor Trajectories**: Displays bounding boxes for dynamic objects
- **Scene Bounding Box**: Visualizes the overall scene bounds
- **Interactive Controls**: Toggle visibility of different elements via UI

## Usage

### Basic Usage

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --agent-id agent_1
```

### With Custom Frame Range

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --agent-id agent_1 \
    --start-frame 1 \
    --end-frame 10
```

### Specific Cameras

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --agent-id agent_1 \
    --cameras camera_1 camera_2
```

### Custom Port

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --port 8080
```

### Larger Image Thumbnails

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --image-thumbnail-size 200
```

### All Options

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py --help
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | Path | (required) | Directory containing simulator data |
| `--agent-id` | str | "agent_1" | ID of the agent to load |
| `--start-frame` | int | 1 | Start frame number |
| `--end-frame` | int | None | End frame number (None = all frames) |
| `--cameras` | tuple | ("camera_1", ..., "camera_4") | Which cameras to visualize |
| `--lidars` | tuple | ("lidar_1",) | Which lidars to visualize |
| `--camera-frustum-scale` | float | 0.5 | Scale of camera frustums |
| `--image-thumbnail-size` | int | 100 | Size of image thumbnails in pixels |
| `--point-size` | float | 0.02 | Size of lidar points |
| `--max-points-per-cloud` | int | 50000 | Max points per lidar scan (None = all) |
| `--show-trajectories` | bool | True | Whether to show actor trajectories |
| `--port` | int | 7007 | Port for the viser server |
| `--host` | str | "0.0.0.0" | Host for the viser server |

## Interactive Controls

Once the viewer is running, you can:

1. **Navigate**: Click and drag to rotate, scroll to zoom
2. **Toggle Visibility**: Use the checkboxes in the UI panel to show/hide:
   - Cameras
   - LiDAR point clouds
   - Actor trajectories
3. **Jump to Camera**: Click on any camera frustum to move the view to that camera's position

## Troubleshooting

### Port Already in Use

If you get a port error, specify a different port:

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --port 8080
```

### Too Many Points

If the visualization is slow, reduce the number of points:

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --max-points-per-cloud 10000
```

### Memory Issues

Try loading fewer frames:

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --start-frame 1 \
    --end-frame 5
```

## Verification Checklist

Use this tool to verify:

- [ ] Camera poses are at expected locations
- [ ] Camera orientations point in the right direction
- [ ] Camera images are displayed correctly in frustums
- [ ] Image content matches the expected view direction
- [ ] LiDAR poses align with cameras
- [ ] Point clouds are correctly positioned in world space
- [ ] Point clouds align with visible scene content in camera images
- [ ] Point cloud intensity values are reasonable
- [ ] Actor trajectories align with the scene
- [ ] Scene bounding box encompasses all data
- [ ] Multiple cameras have consistent poses over time

## Technical Details

### Coordinate Systems

The visualizer displays data in the NeRF Studio coordinate system:
- +X: Right
- +Y: Up  
- +Z: Forward (into the scene)

### Point Cloud Colors

LiDAR points are colored by intensity values:
- Dark blue/purple: Low intensity
- Yellow/white: High intensity

### Actor Trajectory Colors

Different object types have different colors:
- Green shades: Vehicles (cars)
- Red shades: Heavy vehicles (trucks, buses)
- Cyan: Motorbikes
- Magenta: Pedestrians
- Blue shades: Bicycles/Cyclists
- Gray: Unknown types

## Examples

### Quick Check for First 10 Frames

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --end-frame 10 \
    --max-points-per-cloud 20000
```

### Debug Specific Camera

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --cameras camera_1 \
    --end-frame 5
```

### Full Scene Visualization

```bash
python nerfstudio/scripts/viewer/visualize_dataparser.py \
    --data /path/to/simulator/data \
    --max-points-per-cloud 100000 \
    --camera-frustum-scale 1.0
```

