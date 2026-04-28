import cv2

def main():
    # Initialize camera using V4L2 backend
    # Change 0 to your camera index if multiple cameras are connected
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Target parameters: MJPEG 1280x960 @ 30 FPS
    width, height = 1280, 960
    fps = 30

    # Set the FourCC code to MJPG (Motion JPEG)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set Frame Rate
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Verification: Read back settings to ensure the hardware accepted them
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"--- Camera Configuration ---")
    print(f"Requested: {width}x{height} @ {fps} FPS (MJPG)")
    print(f"Actual:    {int(actual_w)}x{int(actual_h)} @ {actual_fps} FPS ({fourcc_str})")
    print(f"-----------------------------")
    print("Press 'q' to exit.")

    window_name = "Camera Functionality Tester"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the camera feed
        cv2.imshow(window_name, frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
