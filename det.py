# Step 1: Install dependencies (run this in Colab)

# Step 2: Import libraries and configure Detectron2
import torch
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

# Configuration for the Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Change if using a different model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detection
cfg.MODEL.WEIGHTS = r"C:\Users\HP\Downloads\model_final.pth"


# Initialize the Detectron2 predictor
predictor = DefaultPredictor(cfg)

# Step 3: Capture frames from IP webcam and perform detection
def main():
    # Get IP webcam URL from the user
    ip_webcam_url = input("Enter the IP webcam URL (e.g., http://<IP_ADDRESS>:<PORT>/video): ")

    # Open a connection to the IP webcam
    cap = cv2.VideoCapture(ip_webcam_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    else:
        print("Successfully opened video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert the frame to RGB (Detectron2 expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform Detectron2 inference on the frame
        outputs = predictor(frame_rgb)

        # Draw the detection results on the frame
        v = Visualizer(frame_rgb[:, :, ::-1], scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Convert the result to BGR format to display with OpenCV
        result_frame = out.get_image()[:, :, ::-1]

        # Display the frame with detections
        cv2.imshow('Detectron2 Detection', result_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
