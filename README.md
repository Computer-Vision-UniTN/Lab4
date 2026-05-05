# Camera Calibration

## 🧠 What is Camera Calibration?
Camera calibration is a fundamental task in computer vision that aims to estimate the mathematical model that maps 3D scene points to 2D image pixels. This process allows us to recover the geometry of the camera, correct lens distortion, and use images for metric measurements and stereo vision.
Instead of treating the image as a simple 2D picture, calibration allows us to understand how the camera projects the 3D world onto the image plane. This is essential when we want to measure distances, reconstruct 3D information, or align images acquired from multiple cameras.

- **Intrinsic Parameters**: Describe the internal properties of the camera, such as focal length, principal point, and lens distortion coefficients.
- **Extrinsic Parameters**: Describe the pose of the camera with respect to the world or calibration target, using rotation and translation.
- **Chessboard Calibration**: Use a known planar target to build correspondences between 3D points on the board and their detected 2D image coordinates.
- **Image Rectification**: Warp stereo images so that corresponding points lie on the same image row, simplifying stereo matching.

##📝 Note
The code inside the Python code folder is designed to perform live calibration with a stereo camera.
The notebook, instead, uses a pre-recorded video for the calibration process.
