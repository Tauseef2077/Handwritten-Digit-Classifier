from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite

DEBUG_MODE = False  

app = Flask(__name__)
CORS(app)

print("Loading TFLite model...")
interpreter = None

try:
    interpreter = tflite.Interpreter(model_path='mnist_model.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✓ Model loaded successfully!")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'mnist_model.tflite' exists in the same directory")
    interpreter = None

def preprocess_image(image_data, save_debug=False):
    """Preprocess canvas image - For WHITE digits on BLACK background"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        image = image.convert('L')
        image_array = np.array(image)
        
        if save_debug:
            import os
            os.makedirs('debug', exist_ok=True)
            Image.fromarray(image_array).save('debug/1_original.png')
        
        threshold = 50
        rows = np.any(image_array > threshold, axis=1)
        cols = np.any(image_array > threshold, axis=0)
        
        if not rows.any() or not cols.any():
            print(" Empty canvas")
            return np.zeros((1, 28, 28, 1), dtype=np.float32)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        height = ymax - ymin
        width = xmax - xmin
        pad_y = int(height * 0.2)
        pad_x = int(width * 0.2)
        
        ymin = max(0, ymin - pad_y)
        ymax = min(image_array.shape[0], ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(image_array.shape[1], xmax + pad_x)
        
        cropped = image_array[ymin:ymax, xmin:xmax]
        
        max_dim = max(cropped.shape[0], cropped.shape[1])
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        y_offset = (max_dim - cropped.shape[0]) // 2
        x_offset = (max_dim - cropped.shape[1]) // 2
        square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        
        resized = Image.fromarray(square).resize((28, 28), Image.Resampling.LANCZOS)
        resized_array = np.array(resized)
        
        normalized = resized_array.astype(np.float32) / 255.0
        
        final = normalized.reshape(1, 28, 28, 1)
        
        return final
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'MNIST API (TFLite)',
        'model_loaded': interpreter is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': interpreter is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict digit using TFLite model"""
    
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if interpreter is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        processed = preprocess_image(data['image'], save_debug=DEBUG_MODE)
        
        if processed is None:
            return jsonify({'error': 'Image processing failed'}), 400
        
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        probs = predictions[0]
        
        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit])
        
        result = {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': {str(i): float(probs[i]) for i in range(10)}
        }
        
        print(f"Prediction: {predicted_digit} (confidence: {confidence:.2%})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    print("\n" + "="*60)
    print("MNIST Digit Recognition API - TFLite")
    print("="*60)
    print(f"\nModel loaded: {interpreter is not None}")
    
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\nStarting server on port {port}")
    print("\nEndpoints:")
    print("  GET  /         - Status")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Predict digit")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
