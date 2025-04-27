from flask import Flask, render_template, Response, send_file, request, jsonify
from ultralytics import YOLO
import cv2
import time
import os
import threading
import queue
import fpdf
import datetime
from io import BytesIO

import time
import queue
import numpy as np
import cv2
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter

import logging
import numpy as np

import torch
import torch.serialization

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Global variables
camera = None
detection_thread = None
is_running = False
results_queue = queue.Queue()
detected_words = []
output_file = "detected_signs.txt"
stop_event = threading.Event()
speech_active = False
speech_stop_event = threading.Event()
tts_enabled = False

# For real-time updates to frontend
latest_detection = None
detection_time = 0

# We'll use the browser's TTS instead of pyttsx3
# This simplifies our backend code

def load_model():
    # Load YOLO model from Ultralytics
    try:
        model_path = os.path.join(os.getcwd(), "best.pt")
        if not os.path.exists(model_path):
            # If the model isn't in the current directory, try a default path
            model_path = "best.pt"
        
        # Load the model using YOLO class
        model = YOLO(model_path)
        return model
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

def initialize_camera():
    import time
    print("Trying to open camera...")
    
    cam = None
    for attempt in range(10):  # Try up to 10 times
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced from 640
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 640
        
        if cam.isOpened():
            print("Camera opened successfully on attempt", attempt + 1)
            return cam
        else:
            print(f"Attempt {attempt + 1}: Camera not opened, retrying...")
            time.sleep(1)

    raise RuntimeError("Failed to access the camera after several attempts.")


# Add these to your global variables
import time
active_boxes = {}  # Format: {class_name: {"box": (x1,y1,x2,y2), "conf": conf, "expire_time": timestamp}}
BOX_DISPLAY_DURATION = 2.0  # Display boxes for 2 seconds
FRAME_SKIP = 3  # Process every 3rd frame instead of every 2nd
CONFIDENCE_THRESHOLD = 0.5  # Slightly increased

def detection_loop(model):
    global camera, is_running, detected_words, stop_event, tts_enabled, latest_detection, detection_time
    
    camera = initialize_camera()
    
    # More optimized parameters for accuracy
    CONFIDENCE_THRESHOLD = 0.55  # Increased confidence threshold
    MIN_DETECTION_FRAMES = 2  # Reduced for faster response
    DETECTION_COOLDOWN = 0.3  # Faster detection cycle
    
    # Detection tracking variables
    detection_counts = {}  # Track consecutive detections
    last_detection_time = 0
    last_detected = None
    
    # State variables for persistent box display
    persistent_boxes = {}  # {class_name: {"box": (x1,y1,x2,y2), "conf": conf, "until": timestamp}}
    BOX_PERSISTENCE = 1.5  # Show boxes for 1.5 seconds
    
    frame_count = 0
    
    while not stop_event.is_set():
        success, frame = camera.read()
        if not success:
            break
            
        current_time = time.time()
        frame_count += 1
        
        # Only process every 4th frame to reduce CPU load
        process_this_frame = frame_count % 4 == 0
        
        # Clear expired persistent boxes
        for class_name in list(persistent_boxes.keys()):
            if current_time > persistent_boxes[class_name]["until"]:
                del persistent_boxes[class_name]
        
        # Run detection only on selected frames
        if process_this_frame:
            # Run detection on the frame
            results = model(frame)
            
            # Process results
            current_frame_detections = {}
            
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    # Get the class name and confidence
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    conf = float(box.conf[0])
                    
                    if conf > CONFIDENCE_THRESHOLD:
                        # Track in current frame detections
                        current_frame_detections[class_name] = conf
                        
                        # Add to persistent boxes for display
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        persistent_boxes[class_name] = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "until": current_time + BOX_PERSISTENCE
                        }
            
            # Update detection counts with smoother decay
            for class_name in list(detection_counts.keys()):
                if class_name not in current_frame_detections:
                    # Gradual decay for stability
                    detection_counts[class_name] -= 0.3
                    if detection_counts[class_name] <= 0:
                        del detection_counts[class_name]
                else:
                    # Increment with confidence weighting
                    detection_counts[class_name] += 0.7 + (current_frame_detections[class_name] * 0.3)
            
            # Add new detections to counts
            for class_name in current_frame_detections:
                if class_name not in detection_counts:
                    detection_counts[class_name] = 1.0
            
            # Check for stable detections
            stable_detections = [cls for cls, count in detection_counts.items() 
                               if count >= MIN_DETECTION_FRAMES]
            
            # Process stable detections
            if stable_detections and (current_time - last_detection_time) > DETECTION_COOLDOWN:
                stable_detections.sort(key=lambda cls: detection_counts[cls], reverse=True)
                # Take only the most confident detection
                stable_detections = stable_detections[:1]
                    
                detection_str = ", ".join(stable_detections)
                
                # Only record if different from last detected
                if detection_str != last_detected:
                    with open(output_file, "a") as f:
                        f.write(f"{detection_str}\n")
                    
                    detected_words.extend(stable_detections)
                    
                    # Update latest detection for TTS
                    if tts_enabled:
                        latest_detection = detection_str
                        detection_time = current_time
                        print(f"New detection: {detection_str}")
                    
                    # Partial reset of counter for stability
                    for class_name in stable_detections:
                        detection_counts[class_name] = 1.5
                    
                    last_detection_time = current_time
                    last_detected = detection_str
        
        # Draw all persistent boxes on every frame for consistent display
        for class_name, box_info in persistent_boxes.items():
            x1, y1, x2, y2 = box_info["box"]
            conf = box_info["conf"]
            
            # Draw bounding box with high visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add text background for better visibility
            text = f"{class_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Put the frame in the queue for the generator
        try:
            results_queue.put_nowait(frame_bytes)  # Use non-blocking put
        except queue.Full:
            # If queue is full, skip this frame
            pass
        
        # Small sleep to reduce CPU usage
        time.sleep(0.01)
    
    # Release camera when done
    if camera is not None:
        camera.release()


def draw_persistent_boxes(frame, active_boxes, current_time):
    """Draw all active boxes on the frame and remove expired ones"""
    # Create a copy of the frame to avoid modifying the original
    frame_copy = frame.copy()
    
    # Remove expired boxes
    expired_keys = []
    for class_name, box_info in active_boxes.items():
        if box_info["expire_time"] < current_time:
            expired_keys.append(class_name)
    
    for key in expired_keys:
        del active_boxes[key]
    
    # Draw remaining active boxes
    for class_name, box_info in active_boxes.items():
        x1, y1, x2, y2 = box_info["box"]
        conf = box_info["conf"]
        
        # Draw box with enhanced visibility
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add background to text for better visibility
        text = f"{class_name} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame_copy, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(frame_copy, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    return frame_copy

from fpdf import FPDF

def generate_pdf():
    """Generate a PDF file with all detected signs (including duplicates) from detected_words list"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sign Language Detection Results", ln=True, align="C")
    pdf.ln(10)
    
    # Add detected words
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detected Signs:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    
    if detected_words:
        # Add the detected signs (including duplicates)
        for sign in detected_words:
            pdf.cell(0, 10, sign, ln=True)
    else:
        pdf.cell(0, 10, "No detections recorded", ln=True)
    
    pdf_path = "sign_language_detection_results.pdf"
    pdf.output(pdf_path)
    return pdf_path


@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in your HTML src."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generator function for video streaming."""
    global results_queue
    
    last_frame = None  # Keep track of the last valid frame
    empty_frame_counter = 0  # Counter to track how long we've been sending blank frames
    
    while True:
        try:
            # Get frame from the queue with a 2-second timeout
            frame_bytes = results_queue.get(timeout=2.0)
            last_frame = frame_bytes  # Save the valid frame
            empty_frame_counter = 0  # Reset blank frame counter
            
            # Yield the current valid frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
        except queue.Empty:
            # If queue is empty and no previous valid frame, yield a blank frame
            if last_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n\r\n')
            else:
                # After multiple empty frames, yield a blank frame with some logic to avoid too many
                if empty_frame_counter < 5:
                    # Send blank frame up to 5 times consecutively
                    blank_frame = np.ones((320, 320, 3), dtype=np.uint8) * 200  # Light gray frame
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
                    empty_frame_counter += 1
                else:
                    # After 5 empty frames, pause briefly to avoid continuous retries
                    time.sleep(0.1)
                    empty_frame_counter = 0  # Reset after pause
        
        except Exception as e:
            app.logger.error(f"Error in generate_frames: {e}")
            time.sleep(0.1)  # Brief pause on error
            
@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    """Generate and download the PDF with detection results."""
    try:
        pdf_path = generate_pdf()
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_thread, is_running, detected_words, stop_event, results_queue
    try:
        print("Start detection triggered!")
        
        if is_running:
            return jsonify({"status": "warning", "message": "Detection already running"})
        
        # Immediately send response that we're starting (don't wait for camera init)
        is_running = True
        
        # Clear previous state
        detected_words = []
        stop_event.clear()
        
        # Empty the queue
        while not results_queue.empty():
            try:
                results_queue.get_nowait()
            except:
                break
        
        # This ensures the video feed shows something during initialization
        def send_init_frames():
            for _ in range(10):  # Send a few frames
                init_frame = np.ones((320, 320, 3), dtype=np.uint8) * 200  # Light gray
                cv2.putText(init_frame, (30, 160), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                _, buffer = cv2.imencode('.jpg', init_frame)
                results_queue.put(buffer.tobytes())
                time.sleep(0.2)
        
        # Start the camera initialization in a separate thread
        threading.Thread(target=send_init_frames, daemon=True).start()
        
        # Start the actual detection thread with a timeout
        def start_detection_with_timeout():
            try:
                # Load model with timeout handling
                model = load_model()
                detection_thread = threading.Thread(target=detection_loop, args=(model,))
                detection_thread.daemon = True
                detection_thread.start()
                print("Detection started.")
            except Exception as e:
                app.logger.error(f"Error in camera initialization: {e}")
                # If we fail, set running to false
                global is_running
                is_running = False
                # Send error frame
                error_frame = np.ones((320, 320, 3), dtype=np.uint8) * 200
                cv2.putText(error_frame, "Camera initialization failed", (20, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(error_frame, "Please try again", (80, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                results_queue.put(buffer.tobytes())
        
        # Start the actual detection with timeout in another thread
        threading.Thread(target=start_detection_with_timeout, daemon=True).start()
        
        return jsonify({"status": "success", "message": "Initializing camera..."})
        
    except Exception as e:
        app.logger.error(f"Error starting detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_running, stop_event, camera
    try:
        print("Stop detection triggered!")
        
        if not is_running:
            return jsonify({"status": "warning", "message": "Detection not running"})
        
        # Signal thread to stop
        stop_event.set()
        is_running = False
        
        # Force camera release to ensure clean shutdown
        if camera:
            try:
                camera.release()
            except:
                pass
        
        print("Detection stopped.")
        return jsonify({"status": "success", "message": "Detection stopped"})
    
    except Exception as e:
        app.logger.error(f"Error stopping detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/toggle_tts', methods=['POST'])
def toggle_tts():
    global tts_enabled
    try:
        # Flip the toggle
        tts_enabled = not tts_enabled
        print(f"TTS Enabled: {tts_enabled}")

        # Send updated status back to frontend
        return jsonify({
            "status": "success",
            "tts_enabled": tts_enabled
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get_latest_detection', methods=['GET'])
def get_latest_detection():
    """Endpoint for the frontend to poll for new detections to speak"""
    global latest_detection, detection_time
    
    try:
        # Only return a detection if it's recent (within last 3 seconds)
        current_time = time.time()
        if latest_detection and (current_time - detection_time) < 3:
            detection = latest_detection
            # Clear it so we don't repeat
            latest_detection = None
            return jsonify({
                "status": "success",
                "detection": detection,
                "speak": tts_enabled
            })
        else:
            return jsonify({
                "status": "no_detection"
            })
    except Exception as e:
        app.logger.error(f"Error getting latest detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/speak_pdf', methods=['POST'])
def speak_pdf():
    global speech_active, speech_stop_event
    
    try:
        # Get text to speak
        text = ""
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                text = f.read()
        else:
            text = "No signs detected to read out."
        
        # Set speech as active
        speech_active = True
        speech_stop_event.clear()
        
        # Return text for browser-based speech
        return jsonify({
            "status": "success", 
            "text": text,
            "speech_active": True
        })
    except Exception as e:
        app.logger.error(f"speak_pdf error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_speech', methods=['POST'])
def stop_speech():
    global speech_active, speech_stop_event
    
    try:
        speech_stop_event.set()
        speech_active = False
        return jsonify({"status": "success", "message": "Speech stopped"})
    except Exception as e:
        app.logger.error(f"stop_speech error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/get_detection_status', methods=['GET'])
# def get_detection_status():
#     global is_running, detected_words, speech_active, tts_enabled
    
#     try:
#         return jsonify({
#             "is_running": is_running,
#             "detected_count": len(detected_words) if detected_words else 0,
#             "speech_active": speech_active,
#             "tts_enabled": tts_enabled
#         })
#     except Exception as e:
#         app.logger.error(f"Error fetching status: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_detection_status', methods=['GET'])
def get_detection_status():
    global is_running, detected_words, speech_active, tts_enabled
    
    try:
        return jsonify({
            "is_running": is_running,
            "detected_count": len(detected_words),  # Unique count
            "speech_active": speech_active,
            "tts_enabled": tts_enabled
        })
    except Exception as e:
        app.logger.error(f"Error fetching status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

