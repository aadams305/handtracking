import cv2

# Raw camera test (this file). For SimCC student only (ONNX or PyTorch), same V4L2/MJPG style as below:
#   PYTHONPATH=. python3 camera_student.py --backend onnx
#   PYTHONPATH=. python3 camera_student.py --backend pytorch --device cuda
#
# Optional: MediaPipe teacher or student via package:
#   PYTHONPATH=. python3 -m handtracking.live_camera --source teacher
#   PYTHONPATH=. python3 -m handtracking.live_camera --source student --backend onnx --infer-every 3

def main():
    # Initialize camera using V4L2 backend as in the original script
    # Change 0 to your camera index if multiple cameras are connected
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Matching the specific parameters from your provided script
    width, height = 1280, 960
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    window_name = "Camera Functionality Tester"
    print(f"Starting camera at {width}x{height} @ 60FPS...")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the raw camera feed
        cv2.imshow(window_name, frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
