# app.py - Flask API
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from datetime import datetime
import uuid
import os

app = Flask(__name__)
CORS(app)

# Initialize detector
from train_microplastics_model import MicroplasticDetector
detector = MicroplasticDetector()

# Load your trained model
try:
    detector.load_trained_model('./runs/detect/microplastics_detector/weights/best.pt')
except:
    print("Model not found. Please train the model first.")

@app.route('/api/detect', methods=['POST'])
def detect_microplastics():
    """API endpoint for microplastic detection"""
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Save temporary image
        temp_path = f"temp_{uuid.uuid4()}.jpg"
        image.save(temp_path)
        
        # Get detection results
        detections, annotated_image = detector.detect_microplastics(temp_path)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate statistics
        total_particles = len(detections)
        particle_types = {}
        for detection in detections:
            particle_type = detection['class']
            particle_types[particle_type] = particle_types.get(particle_type, 0) + 1
        
        # Clean up
        os.remove(temp_path)
        
        response = {
            'success': True,
            'total_particles': total_particles,
            'particle_types': particle_types,
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}",
            'timestamp': datetime.now().isoformat(),
            'location': data.get('location', None)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/simulate', methods=['POST'])
def simulate_detection():
    """Simulate sensor data input for hackathon demo"""
    try:
        data = request.get_json()
        
        # Simulate realistic microplastic readings
        simulated_data = {
            'success': True,
            'total_particles': np.random.randint(10, 100),
            'particle_types': {
                'fiber': np.random.randint(2, 20),
                'fragment': np.random.randint(5, 30),
                'sphere': np.random.randint(1, 15),
                'microplastic': np.random.randint(3, 25)
            },
            'concentration': round(np.random.uniform(0.1, 5.0), 2),
            'water_quality_score': round(np.random.uniform(3.0, 9.0), 1),
            'dominant_plastic_type': np.random.choice(['PE', 'PP', 'PET', 'PS', 'PVC']),
            'timestamp': datetime.now().isoformat(),
            'location': data.get('location', {'lat': 40.7128, 'lng': -74.0060})
        }
        
        return jsonify(simulated_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': detector.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
