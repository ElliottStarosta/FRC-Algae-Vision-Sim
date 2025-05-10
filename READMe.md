# Robot-Algae Interception Simulator

A physics-based simulation that demonstrates interception between a robot (blue) and a moving algae target (green). This simulator visualizes various interception scenarios considering acceleration constraints, maximum velocity, deceleration effects, and predictive targeting.



## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/robot-algae-simulator.git
cd robot-algae-simulator

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
python simulator.py
```

## Controls

### General Controls

| Key/Action | Description |
|------------|-------------|
| `1` | Normal case scenario |
| `2` | Algae comes to rest before interception |
| `3` | Robot doesn't reach max velocity |
| `4` | Algae moving in front of robot |
| `5` | Algae traveling faster than robot's max velocity |
| `6` | Robot's initial velocity is negative |
| `7` | Enter playground mode (custom setup) |
| `Space` | Pause/resume simulation |
| `R` | Reset current case |
| `Backspace` | Restore previous state |

### Playground Mode Controls

The playground mode allows you to set up custom interception scenarios through the following steps:

1. **Place robot**: Click to position the robot
2. **Set robot velocity**: Click and drag to set direction and magnitude
3. **Place algae**: Click to position the algae
4. **Set algae velocity**: Click and drag to set direction and magnitude
5. **Adjust parameters**:
   - `Up/Down` arrows: Increase/decrease robot acceleration
   - `Left/Right` arrows: Decrease/increase robot max velocity
   - `A/Z` keys: Increase/decrease algae deceleration
   - `Enter`: Start simulation
   - `Esc`: Cancel and return to normal mode

### Object Manipulation (during setup)

After completing the initial setup in playground mode:
- **Right-click** on the robot or algae to select and move them
- **Right-click** on velocity vectors to adjust them
- When an object is selected, you can drag it to reposition it

## Test Cases

### 1. Normal Case
Standard interception scenario with default parameters.

### 2. Algae Comes to Rest
The algae decelerates and eventually stops. The robot must predict the final resting position.

### 3. Robot Doesn't Reach Max Velocity
Interception occurs before the robot reaches its maximum velocity constraint.

### 4. Algae Moving in Front of Robot
The algae starts in front of the robot and moves away. The robot must catch up.

### 5. Fast Algae
The algae moves faster than the robot's maximum velocity, making direct pursuit impossible.

### 6. Negative Initial Velocity
The robot starts moving away from the algae and must reverse direction.

### 7. Playground Mode
Create your own custom scenarios with adjustable parameters.
