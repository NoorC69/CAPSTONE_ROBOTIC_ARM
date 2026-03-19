from flask import Flask, render_template, jsonify
import numpy as np

app = Flask(__name__)

# Arm lengths in cm
L = [50.0, 50.0, 25.0]

# Joint angles for 5 DOF arm
q = np.zeros(5)

# CAPTURE STATE
captured = False
capture_offset = np.zeros(3)  # Offset from arm tip to target center when captured

t = 0.0

# ---------------- FORWARD KINEMATICS ----------------
def forward(q):
    pts = [np.array([0.0, 0.0, 0.0])]
    T = np.eye(3)

    # Shoulder (link 0): yaw and pitch
    yaw = q[0]
    pitch = q[1]
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    T = T @ Ry @ Rx
    pts.append(pts[-1] + T @ np.array([0, L[0], 0]))

    # Elbow (link 1): yaw and pitch
    yaw = q[2]
    pitch = q[3]
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    T = T @ Ry @ Rx
    pts.append(pts[-1] + T @ np.array([0, L[1], 0]))

    # Wrist (link 2): only pitch
    yaw = 0.0
    pitch = q[4]
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    T = T @ Ry @ Rx
    pts.append(pts[-1] + T @ np.array([0, L[2], 0]))

    return pts

# ---------------- IK STEP ----------------
def ik_step(q, target, max_iterations=3, damping=0.01):
    pts = forward(q)
    end = pts[-1]
    error = target - end

    if np.linalg.norm(error) < 0.5:
        return q

    J = np.zeros((3, len(q)))
    eps = 1e-3

    for i in range(len(q)):
        dq_pert = np.zeros_like(q)
        dq_pert[i] = eps
        end_pert = forward(q + dq_pert)[-1]
        J[:, i] = (end_pert - end) / eps

    rcond = damping
    
    try:
        J_pinv = np.linalg.pinv(J, rcond=rcond)
        dq = 0.5 * J_pinv @ error
    except np.linalg.LinAlgError:
        dq = 0.02 * J.T @ error

    dq = np.clip(dq, -0.1, 0.1)
    new_q = np.clip(q + dq, -np.pi, np.pi)

    if max_iterations > 1:
        return ik_step(new_q, target, max_iterations - 1, damping)
    else:
        return new_q

# ---------------- ROUTE ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/step")
def step():
    global q, t, captured, capture_offset

    t += 0.01

    # Chaser satellite (with arm)
    sat1 = np.array([100 * np.cos(t), 80, 100 * np.sin(t)])

    # Get current arm end position in world coordinates
    local_pts = forward(q)
    arm_end_world = sat1 + local_pts[-1]

    if not captured:
        # Normal operation: target follows its orbit
        sat2 = np.array([
            150 * np.cos(t + 1),
            100,
            150 * np.sin(t + 1)
        ])
        
        # Transform target into satellite frame for IK
        target_local = sat2 - sat1
        q = ik_step(q, target_local, max_iterations=3)
        
        # Check for capture
        dist = np.linalg.norm(arm_end_world - sat2)
        if dist < 10:  # Capture threshold
            captured = True
            # Store the offset from arm tip to target center when captured
            capture_offset = sat2 - arm_end_world
            print(f"CAPTURED! Distance: {dist:.2f}")
    
    else:
        # CAPTURED STATE: Target satellite follows the arm
        sat2 = arm_end_world + capture_offset
        
        # Optional: Add small random movement to simulate "grappling"
        sat2 += np.random.normal(0, 0.1, 3)
        
        # Keep the arm frozen at capture position (no more IK solving)
        # OR allow manual control via additional endpoints

    # Recalculate arm points for rendering
    local_pts = forward(q)
    world_pts = [sat1 + p for p in local_pts]
    
    dist = np.linalg.norm(world_pts[-1] - sat2)

    return jsonify({
        "arm": [p.tolist() for p in world_pts],
        "sat1": sat1.tolist(),
        "sat2": sat2.tolist(),
        "captured": bool(captured),
        "distance": float(dist)
    })

@app.route("/release")
def release():
    """Manual release endpoint"""
    global captured
    captured = False
    print("RELEASED!")
    return jsonify({"status": "released"})

if __name__ == "__main__":
    app.run(debug=True)
