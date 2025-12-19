import numpy as np
import cv2
import json

# --- CONFIGURATION ---
CHECKERBOARD = (15, 10)
SQUARE_SIZE = 12.7  # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

print("Capturing for Logitech C922 @ 1080p")
print("Press 's' to capture. Press 'q' to calculate.")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_corners)

    cv2.imshow('Calibration', display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and ret_corners:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"Captured image {len(imgpoints)}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(imgpoints) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # --- CALCULATE REPROJECTION ERROR ---
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Image {i+1} error: {error:.4f} pixels")

    mean_error = total_error / len(objpoints)
    print(f"\nTotal Mean Reprojection Error: {mean_error:.4f} pixels")
    
    # Save results
    results = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "reprojection_error": mean_error
    }
    with open("c922_calibration.json", "w") as f:
        json.dump(results, f, indent=4)
