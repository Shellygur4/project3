
# Facial Feature-Preserving Filters: Enhancing Facial Expressions in Filtered Videos

## Project Description
  
This project aims to develop a script that applies visual filters to videos of human faces while preserving the core features of facial expressions. The primary goal is to maintain the subtle nuances of facial movements, even when applying transformative filters. This ensures that the original expressions remain recognizable and expressive.

## Problem Statement
Traditional filters often distort or mask the fine details of facial expressions, which are crucial for conveying emotions. This project addresses the challenge of creating filters that maintain the integrity of these delicate movements, ensuring that the original expressions remain discernible.

## Objectives and Goals
Develop a filter script that can be applied to videos, preserving facial expressions.
Create at least three different filter options:
Transforming the face into a cat
Giving a more childlike appearance
Providing a cartoonish look
## Tools and Technologies
Python: The primary programming language used for development.
MediaPipe: Specifically, the Face Mesh module is used for facial landmark detection and stylization.
OpenCV: Used for image and video processing.
## Data and Resources
Sample Videos: Provided for testing and demonstrating the filters developed.
Image Files: Various face images used for creating cartoon filters.
## Installation and Setup

```git clone https://github.com/Shellygur4/project3.git```

Navigate to the Project Directory:


```cd C:\Users\shell\homweork\project\project3```

# Install Dependencies:


``` pip install -r requirements.txt```

## Ensure the MediaPipe Model (face_stylizer.task) is available at the specified path in the code.

# Usage
## for identification of cat person face images and adds a smile to them

place your input image in the project3\different filter useage\cat_person_mesh\data

run the image processing script :

```project3\different filter useage\cat_person_mesh\main.py```

## For chillike filter

place your input image in the project3\different filter useage\child_filter\data

run the image processing script :
```project3\different filter useage\Child_Filter\main.py```

## for a cartoon filter to a single image:

place your input image in the project3\different filter useage\image_to_cartoon_image\data

Run the image processing script:

```project3\different filter useage\image_to_cartoon_image\main.py```

## For Video to cartoon frames
place your input image in the project3\different filter useage\video_to_cartoon_frames\data

Run the video processing script:

```project3\different filter useage\video_to_cartoon_frames\main.py```

## For Video to cartoon video
place your input image in the project3\different filter useage\video_to_cartoon_video\data

Run the video processing script:
```project3\different filter useage\video_to_cartoon_video\main.py```


## Expected Deliverables
GitHub Repository: The project's codebase.

Final Presentation: Output should include filtered videos\images.

## Potential Impact
This project has significant implications for enhancing social interaction skills in children with autism. By creating filters that preserve facial expressions, the tool aims to help autistic children recognize and mimic facial expressions, potentially improving their social interaction skills.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Ensure that all code adheres to the project's coding standards and includes appropriate test coverage.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or additional information, please contact:

Project Lead: Shelly Gur

Email: shellygur4@gmail.com

GitHub: Shellygur4
