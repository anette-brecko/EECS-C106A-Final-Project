# --- CONFIGURATION ---
import numpy as np
import cv2
import glob
import json
from pathlib import Path

# --- CONFIGURATION ---
CHECKERBOARD = (15, 10)
SQUARE_SIZE = 12.7  # mm
IMAGE_FOLDER = Path('/Users/jhaertel/Pictures/board/')

# Termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] 
imgpoints = [] 

# Get list of images
images = list(IMAGE_FOLDER.glob("*.jpg"))
print(images)
if not images:
    print(f"No images found in {IMAGE_FOLDER}. Check the path and file extension.")
    exit()

print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Optional: Draw and display for verification
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Processing Image', img)
        cv2.waitKey(100) # Quick preview
    else:
        print(f"Corners not found in image: {fname}")

cv2.destroyAllWindows()

# --- CALIBRATION ---
if len(imgpoints) > 0:
    print("\nCalculating calibration...")
    # Use the shape of the last processed gray image
    h, w = gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # --- REPROJECTION ERROR ---
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    
    print("\n--- RESULTS ---")
    print(f"Reprojection Error: {mean_error:.4f} pixels")
    print("Camera Matrix (K):")
    print(mtx)
    print("\nDistortion Coefficients (D):")
    print(dist)

    # Save to JSON
    output_data = {
        "resolution": [w, h],
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "reprojection_error": mean_error
    }
    with open("c922_offline_params.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("\nResults saved to c922_offline_params.json")
else:
    print("Could not find corners in any images. Check your CHECKERBOARD dimensions.")      
    json.dump(results, f, indent=4)
