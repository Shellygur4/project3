import cv2
import os
from face_mesh_setup import initialize_face_mesh
from image_processing import draw_face_mesh, modify_smile

# Initialize the MediaPipe Face Mesh solution
face_mesh = initialize_face_mesh()

# Paths for input images and output directories
path = "C:/Users/micha/Documents/PythonHW/project3-2/project3/src/project3/cat_person_mesh/data"
path_out_mesh = "C:/Users/micha/Documents/PythonHW/project3-2/project3/src/project3/cat_person_mesh/output/OutputFaceMesh"
path_out_smile = "C:/Users/micha/Documents/PythonHW/project3-2/project3/src/project3/cat_person_mesh/output/OutputSmile"

# Create directories if they don't exist
os.makedirs(path_out_mesh, exist_ok=True)
os.makedirs(path_out_smile, exist_ok=True)

# Indices for outer and inner lips
outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 324, 318]
smile_factor = 0.5

# Process images
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    frame = cv2.imread(file_path)

    if frame is None:
        print(f"Error loading image {file_path}")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        print(f"No face landmarks detected in {filename}")
        continue

    face_landmarks = results.multi_face_landmarks[0]

    # **Save the original frame for smile modification**
    frame_for_smile = frame.copy()

    # Draw face mesh on the original frame and save the image
    draw_face_mesh(frame, face_landmarks)
    new_name_mesh = os.path.join(path_out_mesh, filename)
    print(f"Saving face mesh image to: {new_name_mesh}")
    if not cv2.imwrite(new_name_mesh, frame):
        print(f"Error saving image with mesh {new_name_mesh}")

    # Modify smile on the copied frame and save the image without the mesh
    modify_smile(frame_for_smile, face_landmarks, smile_factor, outer_lip_indices, inner_lip_indices)
    new_name_smile = os.path.join(path_out_smile, filename)
    print(f"Saving smile image to: {new_name_smile}")
    if not cv2.imwrite(new_name_smile, frame_for_smile):
        print(f"Error saving image with smile {new_name_smile}")

# Release resources
face_mesh.close()
