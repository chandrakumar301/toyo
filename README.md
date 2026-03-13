# Toyota Component Defect Detection System

A Flask-based web application for industrial quality inspection of Toyota car parts using advanced computer vision algorithms.

## Features

- **Scratch Detection**: Automated detection of surface defects (scratches, cracks, dents) on car components
- **Industrial Accuracy**: Optimized algorithms that ignore normal texture, lighting, and noise
- **Web Interface**: User-friendly upload interface with real-time results
- **Confidence Scoring**: Provides confidence levels and defect area percentages
- **Visual Feedback**: Preview images with defect highlighting and bounding boxes

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Computer Vision**: OpenCV for optimized image processing
- **Image Processing**: Pillow (PIL) for image handling
- **Numerical Computing**: NumPy for array operations
- **Frontend**: HTML/CSS/JavaScript with responsive design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chandrakumar301/toyo.git
cd toyo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload an image of a Toyota component (JPG, PNG formats supported)
2. The system analyzes the image for surface defects
3. View results including:
   - Defect status (OK PART or DEFECT)
   - Confidence percentage
   - Defect area percentage
   - Visual preview with defect markers

## API Endpoints

- `GET /`: Main web interface
- `POST /api/detect-scratch`: Single image defect detection
- `POST /api/detect-dataset`: Batch processing of multiple images

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Flask 2.0+
- Pillow 9.0+
- NumPy 1.24+

## License

This project is proprietary software for Toyota component inspection.