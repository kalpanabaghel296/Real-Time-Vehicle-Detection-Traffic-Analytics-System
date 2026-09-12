# Wrong-Way Vehicle Detection & Temporal Windowing: Technical Guide & Whiteboard Concepts

---

## 1. Wrong-Way Detection Pipeline

```
Tracked Vehicle Centroid Trajectory Queue [(x_1, y_1), (x_2, y_2), ... (x_k, y_k)]
       ↓
Motion Vector Estimation (Windowed displacement over k frames):
   dx = x_k - x_1
   dy = y_k - y_1
   distance = sqrt(dx^2 + dy^2)
       ↓
Filter: Is distance >= min_movement_distance (e.g. 15 px)?
   ├── NO  → Mark as STATIONARY (reject sub-pixel detector jitter)
   └── YES → Determine dominant motion axis:
             |dy| >= |dx| → "DOWN" (+dy) or "UP" (-dy)
             |dx| >  |dy| → "RIGHT" (+dx) or "LEFT" (-dx)
       ↓
Compare with Configured Allowed Flow (e.g., Allowed: "DOWN"):
   ├── Matches Flow → Reset / decay violation counter
   └── Opposes Flow → Increment veh.violation_frames += 1
       ↓
Temporal Confirmation Gate: Has veh.violation_frames >= confirm_frames (e.g., 4 frames)?
   ├── NO  → Pending confirmation (avoids single-frame false alarms)
   └── YES & not yet alerted → CONFIRMED WRONG-WAY VIOLATION!
       ↓
Automated Event Dispatch:
   ├── Save annotated violation snapshot frame with red warning bounding box
   ├── Render flashing [!] WRONG-WAY ALERT [!] HUD banner
   └── Append structured event record to audit log
```

---

## 2. Core Concepts: WHAT, WHY, and HOW

### A. What is Wrong-Way Vehicle Detection?
* **WHAT**: An automated video analytics safety system that identifies vehicles traveling counter to the designated legal flow of traffic on one-way streets, divided highways, or freeway off-ramps.
* **WHY**: Wrong-way driving (WWD) collisions on high-speed highways result in disproportionately severe head-on fatalities. Immediate automated detection (within 1–2 seconds) allows intelligent transportation systems (ITS) to alert oncoming motorists, illuminate electronic warning signs, and notify law enforcement before a collision occurs.

---

### B. Why Single-Frame Direction Decision Causes False Positives (Detector Jitter)
* **The Whiteboard Trap**:
  A naive implementation checks direction simply as:
  $$\Delta y = y_t - y_{t-1}$$
  If $\Delta y < 0$, raise alarm immediately!
* **Why this Fails in Real Life**:
  1. **Bounding Box Scale Jitter**: Neural network detectors exhibit 1–3 pixel variations between frames even on stationary parked vehicles due to quantization, compression artifacts, and lighting fluctuations.
  2. **Lane Changes & Swerving**: A vehicle legitimately traveling DOWN may swerve slightly left or right to avoid a pothole, momentarily generating an unexpected displacement vector for 1 frame.
  3. **Perspective foreshortening**: Small movements in distant camera views translate to noisy pixel deltas.

* **Our Two-Tier Solution**:
  1. **Spatial Filtering (`min_movement_distance`)**: Vehicles must accumulate at least $15\text{ pixels}$ of net spatial displacement across the window before a cardinal direction is assigned; otherwise, the vehicle is classified as `STATIONARY`.
  2. **Temporal Confirmation Window (`confirm_frames`)**: The vehicle must violate the traffic rule for $N$ consecutive frames (e.g., 4–5 frames, $\approx 0.3\text{--}0.5$ seconds) before an official alert is triggered. If the direction normalizes within that window, the violation counter decays and no false alarm is dispatched.

---

### C. Image Coordinate Convention vs. World Heading
$$\begin{matrix}
\text{Movement Direction} & \Delta x & \Delta y \\
\hline
\textbf{DOWN} & \text{negligible} & > 0 \\
\textbf{UP} & \text{negligible} & < 0 \\
\textbf{RIGHT} & > 0 & \text{negligible} \\
\textbf{LEFT} & < 0 & \text{negligible}
\end{matrix}$$

* **Placement Honesty Disclosure**:
  Our system operates on **2D image plane coordinates**, not 3D world coordinates or GPS compass headings. If a road curves sharply in a 2D camera view, physical vehicle heading does not map linearly to screen $(x, y)$. For straight highway corridors, image-plane vector estimation is fast, effective, and requires no complex camera calibration matrix.

---

### D. Automated Snapshot Evidence Capture
When a violation reaches confirmed status:
1. The raw uncompressed BGR frame is copied.
2. A high-contrast red bounding box $(0, 0, 255)$ is drawn around the violator.
3. A top header badge is imprinted with:
   `VIOLATION: ID {track_id} {class} | Moving: {dir} | Allowed: {allowed}`
4. The snapshot is serialized to `outputs/snapshots/violation_id{track_id}_f{frame_idx}.jpg` for forensic audit.

---

## 3. Real-World Limitations & Edge Failure Cases
1. **Curved Roadways (S-Bends)**: On a curving roadway, vehicles driving legally in their lane may temporarily travel "UP" or "LEFT" on screen relative to camera projection. (Solution: Define lane-specific ROI vector corridors).
2. **Camera Perspective Foreshortening**: As vehicles move far into the distance near the horizon, their apparent pixel velocity drops to $< 1\text{ px/frame}$, making direction estimation slower to confirm.
3. **U-Turns**: A vehicle executing a multi-point U-turn produces oscillating directional vectors during the turn maneuver.

---

## 4. Key Placement Interview Questions on Wrong-Way Detection
1. **Why is temporal confirmation preferred over increasing the detection confidence threshold?**
   * *Answer*: Increasing the confidence threshold (e.g. to 0.85) does not eliminate direction noise; an 0.85-confidence car can still have a 2-pixel centroid shift due to box scaling. Temporal confirmation directly targets the *time dimension*, requiring physical persistence of the movement vector across multiple frames.
2. **How would you handle a curved road where traffic direction varies along the lane?**
   * *Answer*: Instead of a single global allowed direction (`DOWN`), we divide the roadway into sequential polygonal zones or lane centerline splines. For each zone, we define a local tangent flow vector $\vec{v}_{\text{allowed}}$. A vehicle's motion vector $\vec{v}_{\text{veh}}$ is evaluated via the vector dot product:
     $$\cos(\theta) = \frac{\vec{v}_{\text{veh}} \cdot \vec{v}_{\text{allowed}}}{\|\vec{v}_{\text{veh}}\| \|\vec{v}_{\text{allowed}}\|}$$
     If $\cos(\theta) < -0.5$ ($\theta > 120^\circ$), a violation is flagged.
3. **What is the latency trade-off of the temporal confirmation buffer?**
   * *Answer*: If video runs at 30 FPS and we require 6 confirmation frames, the alert latency is $6 / 30 = 0.20 \text{ seconds}$. This 200 ms delay is negligible for human operators or warning signs, but eliminates 99%+ of single-frame detector false alarms.
