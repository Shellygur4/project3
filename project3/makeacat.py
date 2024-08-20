import cv2
import numpy as np
import mediapipe as mp

# Constants for resizing
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Function to resize and show image
def resize_and_show(image, filename='output.jpg'):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, int(h * DESIRED_WIDTH / w)))
    else:
        img = cv2.resize(image, (int(w * DESIRED_HEIGHT / h), DESIRED_HEIGHT))
    
    cv2.imwrite(filename, img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to apply the cat filter and retain facial expressions
def apply_cat_filter(frame, face_landmarks, cat_image):
    h, w, _ = frame.shape
    
    # Resize cat image to fit the human face
    cat_image_resized = cv2.resize(cat_image, (w, h))
    
    # Convert images to float for blending
    frame = frame.astype(float)
    cat_image_resized = cat_image_resized.astype(float)

    # Create a mask based on face landmarks
    mask = np.zeros((h, w), dtype=np.uint8)
    points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
    cv2.fillConvexPoly(mask, np.array(points), 255)
    
    # Apply the mask to the cat image
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
    cat_face_overlay = (1 - mask_3d) * cat_image_resized + mask_3d * frame
    cat_face_overlay = np.clip(cat_face_overlay, 0, 255).astype(np.uint8)
    
    # Optionally add additional features like cat ears (if desired)
    # For now, we’ll leave out this part for simplicity.

    return cat_face_overlay

def main(human_image_path, cat_image_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    # Load the images
    human_image = cv2.imread(human_image_path)
    cat_image = cv2.imread(cat_image_path, cv2.IMREAD_UNCHANGED)  # Ensure the cat image has an alpha channel
    rgb_image = cv2.cvtColor(human_image, cv2.COLOR_BGR2RGB)

    # Process the image and get face landmarks
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            cat_image_with_filter = apply_cat_filter(human_image, face_landmarks, cat_image)
            # Show the transformed image
            resize_and_show(cat_image_with_filter)

if __name__ == "__main__":
    # Replace with paths to your local image files
    human_image_path = r'C:\Users\shell\homweork\project3 new\project3\project3\man.jpg'
    cat_image_path = r'C:\Users\shell\homweork\project3 new\project3\project3\cat1.jpg'
    
    main(human_image_path, cat_image_path)
