# from flask import Flask, render_template, Response, send_file, request, jsonify
# from ultralytics import YOLO
# import cv2
# import time
# import os
# import threading
# import queue
# import fpdf
# import datetime

# import logging

# import torch
# import torch.serialization

# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# from flask_cors import CORS
# CORS(app)


# # Global variables
# camera = None
# detection_thread = None
# is_running = False
# results_queue = queue.Queue()
# detected_words = []
# output_file = "detected_signs.txt"
# stop_event = threading.Event()


# import pyttsx3
# tts_engine = pyttsx3.init()


# def load_model():
#     # Load YOLO model from Ultralytics
#     try:
#         model_path = os.path.join(os.getcwd(), "best.pt")
#         if not os.path.exists(model_path):
#             # If the model isn't in the current directory, try a default path
#             model_path = "best.pt"
        
#         # Load the model using YOLO class
#         model = YOLO(model_path)
#         return model
#     except Exception as e:
#         app.logger.error(f"Error loading model: {e}")
#         raise RuntimeError(f"Failed to load model: {e}")


# def initialize_camera():
#     # Initialize camera
#     cam = cv2.VideoCapture(0)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
#     if not cam.isOpened():
#         raise RuntimeError("Could not access the camera.")
#     return cam

# def detection_loop(model):
    
#     global camera, is_running, detected_words, stop_event, tts_enabled 
    
#     camera = initialize_camera()
    
#     # Track the last detected gesture
#     last_detected = set()
    
#     # Open file to save detections
#     with open(output_file, "w") as f:
#         f.write(f"Sign Language Detection Results - {datetime.datetime.now()}\n")
#         f.write("----------------------------------------\n")

    
    
#     while not stop_event.is_set():
#         success, frame = camera.read()
#         if not success:
#             break
        
#         # Run detection on the frame
#         results = model(frame)
        
#         # Process results
#         current_detected = set()
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Get the class name
#                 cls = int(box.cls[0])
#                 class_name = model.names[cls]
#                 conf = float(box.conf[0])
                
#                 if conf > 0.3:  # Confidence threshold
#                     # Add to current frame detections
#                     current_detected.add(class_name)
                    
#                     # Draw bounding box on frame
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{class_name} {conf:.2f}", 
#                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Only record if there's a change in detected gestures
#         if current_detected and current_detected != last_detected:
#             detection_str = ", ".join(current_detected)
#             timestamp = datetime.datetime.now().strftime("%H:%M:%S")
#             with open(output_file, "a") as f:
#                 f.write(f"[{timestamp}] Detected: {detection_str}\n")
#             detected_words.extend(current_detected)
            
#             # Update last detected
#             last_detected = current_detected.copy()
        
#         # Convert frame to JPEG for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
        
#         # Put the frame in the queue for the generator
#         results_queue.put(frame_bytes)
        
#         # Short sleep to reduce CPU usage
#         time.sleep(0.01)
    

#     if current_detected and current_detected != last_detected:
#         detection_str = ", ".join(current_detected)
#         timestamp = datetime.datetime.now().strftime("%H:%M:%S")

#         with open(output_file, "a") as f:
#             f.write(f"[{timestamp}] Detected: {detection_str}\n")
        
#         detected_words.extend(current_detected)
#         last_detected = current_detected.copy()

#         # TTS if enabled
#         if tts_enabled:
#             try:
#                 tts_engine.say(detection_str)
#                 tts_engine.runAndWait()
#             except Exception as e:
#                 app.logger.error(f"TTS Error: {e}")

    
#     # Release camera when done
#     if camera is not None:
#         camera.release()

# def generate_pdf():
#     """Generate a PDF file with the detected signs"""
#     pdf = fpdf.FPDF()
#     pdf.add_page()
    
#     # Set font
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Sign Language Detection Results", ln=True, align="C")
#     pdf.ln(10)
    
#     # Add timestamp
#     pdf.set_font("Arial", "", 12)
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     pdf.cell(0, 10, f"Generated on: {now}", ln=True)
#     pdf.ln(10)
    
#     # Add detected words
#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(0, 10, "Detected Signs:", ln=True)
#     pdf.ln(5)
    
#     pdf.set_font("Arial", "", 12)
    
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             for line in f:
#                 pdf.cell(0, 10, line.strip(), ln=True)
#     else:
#         pdf.cell(0, 10, "No detections recorded", ln=True)
    
#     pdf_path = "sign_language_detection_results.pdf"
#     pdf.output(pdf_path)
#     return pdf_path

# @app.route('/')
# def index():
#     """Return the main page."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in your HTML src."""
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_frames():
#     """Generator function for video streaming."""
#     global results_queue
    
#     while True:
#         try:
#             # Get frame from the queue with a timeout
#             frame_bytes = results_queue.get(timeout=1.0)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
#         except queue.Empty:
#             # If no new frame is available, send a placeholder or wait
#             continue
#         except Exception as e:
#             app.logger.error(f"Error in generate_frames: {e}")
#             break


# @app.route('/download_pdf', methods=['GET'])
# def download_pdf():
#     """Generate and download the PDF with detection results."""
#     try:
#         pdf_path = generate_pdf()
#         return send_file(pdf_path, as_attachment=True)
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})


# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     global detection_thread, is_running, detected_words, stop_event, results_queue
#     try:
#         print("Start detection triggered!")  # Debug log
        
#         if not is_running:
#             # Clear previous results
#             detected_words = []
#             stop_event.clear()
            
#             # Clear any existing items in the queue
#             while not results_queue.empty():
#                 results_queue.get_nowait()
            
#             # Create and start detection thread
#             model = load_model()
#             detection_thread = threading.Thread(target=detection_loop, args=(model,))
#             detection_thread.daemon = True
#             detection_thread.start()
            
#             is_running = True
#             print("Detection started.")
#             return jsonify({"status": "success", "message": "Detection started"})
        
#         return jsonify({"status": "warning", "message": "Detection already running"})

#     except Exception as e:
#         app.logger.error(f"Error starting detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# @app.route('/stop_detection', methods=['POST'])
# def stop_detection():
#     global is_running, stop_event
#     try:
#         print("Stop detection triggered!")  # Debug log
        
#         if is_running:
#             stop_event.set()
#             is_running = False
#             print("Detection stopped.")
#             return jsonify({"status": "success", "message": "Detection stopped"})
        
#         return jsonify({"status": "warning", "message": "Detection not running"})
    
#     except Exception as e:
#         app.logger.error(f"Error stopping detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# tts_enabled = False

# @app.route('/toggle_tts', methods=['POST'])
# def toggle_tts():
#     global tts_enabled
#     try:
#         # Flip the toggle
#         tts_enabled = not tts_enabled

#         # Send updated status back to frontend
#         return jsonify({
#             "status": "success",
#             "tts_enabled": tts_enabled
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# # Add these global variables at the top with your other globals
# speech_active = False
# speech_stop_event = threading.Event()

# @app.route('/speak_pdf', methods=['POST'])
# def speak_pdf():
#     global speech_active, speech_stop_event
    
#     try:
#         # Cancel any ongoing speech
#         if speech_active:
#             speech_stop_event.set()
#             return jsonify({"status": "success", "message": "Speech stopped"})

#         # Start new speech
#         text = ""
#         if os.path.exists(output_file):
#             with open(output_file, 'r') as f:
#                 text = f.read()
#         else:
#             text = "No signs detected to read out."
        
#         # Return text for browser-based speech
#         speech_active = True
#         speech_stop_event.clear()
        
#         return jsonify({
#             "status": "success", 
#             "text": text,
#             "speech_active": True
#         })
#     except Exception as e:
#         app.logger.error(f"speak_pdf error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/stop_speech', methods=['POST'])
# def stop_speech():
#     global speech_active, speech_stop_event
    
#     try:
#         if speech_active:
#             speech_stop_event.set()
#             speech_active = False
#             return jsonify({"status": "success", "message": "Speech stopped"})
        
#         return jsonify({"status": "warning", "message": "No active speech to stop"})
#     except Exception as e:
#         app.logger.error(f"stop_speech error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/get_speech_status', methods=['GET'])
# def get_speech_status():
#     global speech_active
    
#     try:
#         return jsonify({
#             "speech_active": speech_active
#         })
#     except Exception as e:
#         app.logger.error(f"get_speech_status error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# @app.route('/get_detection_status', methods=['GET'])
# def get_detection_status():
#     try:
#         print("Fetching detection status...")  # Debug log
#         return jsonify({
#             "is_running": is_running,
#             "detected_count": len(detected_words) if detected_words else 0
#         })
#     except Exception as e:
#         app.logger.error(f"Error fetching status: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, Response, send_file, request, jsonify
# from ultralytics import YOLO
# import cv2
# import time
# import os
# import threading
# import queue
# import fpdf
# import datetime

# import logging

# import torch
# import torch.serialization

# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# from flask_cors import CORS
# CORS(app)


# # Global variables
# camera = None
# detection_thread = None
# is_running = False
# results_queue = queue.Queue()
# detected_words = []
# output_file = "detected_signs.txt"
# stop_event = threading.Event()
# speech_active = False
# speech_stop_event = threading.Event()


# import pyttsx3
# tts_engine = pyttsx3.init()


# def load_model():
#     # Load YOLO model from Ultralytics
#     try:
#         model_path = os.path.join(os.getcwd(), "best.pt")
#         if not os.path.exists(model_path):
#             # If the model isn't in the current directory, try a default path
#             model_path = "best.pt"
        
#         # Load the model using YOLO class
#         model = YOLO(model_path)
#         return model
#     except Exception as e:
#         app.logger.error(f"Error loading model: {e}")
#         raise RuntimeError(f"Failed to load model: {e}")


# def initialize_camera():
#     print("camera opened")
#     # Initialize camera
#     cam = cv2.VideoCapture(0)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
#     if not cam.isOpened():
#         raise RuntimeError("Could not access the camera.")
#     return cam

# def detection_loop(model):
    
#     global camera, is_running, detected_words, stop_event, tts_enabled
    
#     camera = initialize_camera()
    
#     # Track the last detected gesture
#     last_detected = set()
    
#     # Open file to save detections
#     with open(output_file, "w") as f:
#         f.write(f"Sign Language Detection Results - {datetime.datetime.now()}\n")
#         f.write("----------------------------------------\n")

    
    
#     while not stop_event.is_set():
#         success, frame = camera.read()
#         if not success:
#             break
        
#         # Run detection on the frame
#         results = model(frame)
        
#         # Process results
#         current_detected = set()
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Get the class name
#                 cls = int(box.cls[0])
#                 class_name = model.names[cls]
#                 conf = float(box.conf[0])
                
#                 if conf > 0.3:  # Confidence threshold
#                     # Add to current frame detections
#                     current_detected.add(class_name)
                    
#                     # Draw bounding box on frame
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{class_name} {conf:.2f}", 
#                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         print(current_detectd)
#         # Only record if there's a change in detected gestures
#         if current_detected and current_detected != last_detected:
#             detection_str = ", ".join(current_detected)
#             timestamp = datetime.datetime.now().strftime("%H:%M:%S")
#             with open(output_file, "a") as f:
#                 f.write(f"[{timestamp}] Detected: {detection_str}\n")

#             detected_words.extend(current_detected)
            
#             # Update last detected
#             last_detected = current_detected.copy()
        
#         # Convert frame to JPEG for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
        
#         # Put the frame in the queue for the generator
#         results_queue.put(frame_bytes)
        
#         # Short sleep to reduce CPU usage
#         time.sleep(0.01)
    

#     if current_detected and current_detected != last_detected:
#         detection_str = ", ".join(current_detected)
#         timestamp = datetime.datetime.now().strftime("%H:%M:%S")

#         with open(output_file, "a") as f:
#             f.write(f"[{timestamp}] Detected: {detection_str}\n")
        
#         detected_words.extend(current_detected)
#         last_detected = current_detected.copy()

#         # TTS if enabled
#         if tts_enabled:
#             try:
#                 tts_engine.say(detection_str)
#                 tts_engine.runAndWait()
#             except Exception as e:
#                 app.logger.error(f"TTS Error: {e}")

    
#     # Release camera when done
#     if camera is not None:
#         camera.release()

# def generate_pdf():
#     """Generate a PDF file with the detected signs"""
#     pdf = fpdf.FPDF()
#     pdf.add_page()
    
#     # Set font
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Sign Language Detection Results", ln=True, align="C")
#     pdf.ln(10)
    
#     # Add timestamp
#     pdf.set_font("Arial", "", 12)
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     pdf.cell(0, 10, f"Generated on: {now}", ln=True)
#     pdf.ln(10)
    
#     # Add detected words
#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(0, 10, "Detected Signs:", ln=True)
#     pdf.ln(5)
    
#     pdf.set_font("Arial", "", 12)
    
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             for line in f:
#                 pdf.cell(0, 10, line.strip(), ln=True)
#     else:
#         pdf.cell(0, 10, "No detections recorded", ln=True)
    
#     pdf_path = "sign_language_detection_results.pdf"
#     pdf.output(pdf_path)
#     return pdf_path

# @app.route('/')
# def index():
#     """Return the main page."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in your HTML src."""
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_frames():
#     """Generator function for video streaming."""
#     global results_queue
    
#     while True:
#         try:
#             # Get frame from the queue with a timeout
#             frame_bytes = results_queue.get(timeout=1.0)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
#         except queue.Empty:
#             # If no new frame is available, send a placeholder or wait
#             continue
#         except Exception as e:
#             app.logger.error(f"Error in generate_frames: {e}")
#             break


# @app.route('/download_pdf', methods=['GET'])
# def download_pdf():
#     """Generate and download the PDF with detection results."""
#     try:
#         pdf_path = generate_pdf()
#         return send_file(pdf_path, as_attachment=True)
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})


# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     global detection_thread, is_running, detected_words, stop_event, results_queue
#     try:
#         print("Start detection triggered!")  # Debug log
        
#         if not is_running:
#             # Clear previous results
#             detected_words = []
#             stop_event.clear()
            
#             # Clear any existing items in the queue
#             while not results_queue.empty():
#                 results_queue.get_nowait()
            
#             # Create and start detection thread
#             model = load_model()
#             detection_thread = threading.Thread(target=detection_loop, args=(model,))
#             detection_thread.daemon = True
#             detection_thread.start()
            
#             is_running = True
#             print("Detection started.")
#             return jsonify({"status": "success", "message": "Detection started"})
        
#         return jsonify({"status": "warning", "message": "Detection already running"})

#     except Exception as e:
#         app.logger.error(f"Error starting detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# @app.route('/stop_detection', methods=['POST'])
# def stop_detection():
#     global is_running, stop_event
#     try:
#         print("Stop detection triggered!")  # Debug log
        
#         if is_running:
#             stop_event.set()
#             is_running = False
#             print("Detection stopped.")
#             return jsonify({"status": "success", "message": "Detection stopped"})
        
#         return jsonify({"status": "warning", "message": "Detection not running"})
    
#     except Exception as e:
#         app.logger.error(f"Error stopping detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# tts_enabled = False

# @app.route('/toggle_tts', methods=['POST'])
# def toggle_tts():
#     global tts_enabled
#     try:
#         # Flip the toggle
#         tts_enabled = not tts_enabled

#         # Send updated status back to frontend
#         return jsonify({
#             "status": "success",
#             "tts_enabled": tts_enabled
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/speak_pdf', methods=['POST'])
# def speak_pdf():
#     global speech_active, speech_stop_event
    
#     try:
#         # If speech is active, stop it first
#         if speech_active:
#             speech_stop_event.set()
#             speech_active = False
#             time.sleep(0.1)  # Brief pause to ensure state is updated
        
#         # Get text to speak
#         text = ""
#         if os.path.exists(output_file):
#             with open(output_file, 'r') as f:
#                 text = f.read()
#         else:
#             text = "No signs detected to read out."
        
#         # Set speech as active
#         speech_active = True
#         speech_stop_event.clear()
        
#         # Return text for browser-based speech
#         return jsonify({
#             "status": "success", 
#             "text": text,
#             "speech_active": True
#         })
#     except Exception as e:
#         app.logger.error(f"speak_pdf error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/stop_speech', methods=['POST'])
# def stop_speech():
#     global speech_active, speech_stop_event
    
#     try:
#         if speech_active:
#             speech_stop_event.set()
#             speech_active = False
#             return jsonify({"status": "success", "message": "Speech stopped"})
        
#         return jsonify({"status": "warning", "message": "No active speech to stop"})
#     except Exception as e:
#         app.logger.error(f"stop_speech error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/get_detection_status', methods=['GET'])
# def get_detection_status():
#     global is_running, detected_words, speech_active, tts_enabled
    
#     try:
#         print("Fetching detection status...")  # Debug log
#         return jsonify({
#             "is_running": is_running,
#             "detected_count": len(detected_words) if detected_words else 0,
#             "speech_active": speech_active,
#             "tts_enabled": tts_enabled
#         })
#     except Exception as e:
#         app.logger.error(f"Error fetching status: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, Response, send_file, request, jsonify
# from ultralytics import YOLO
# import cv2
# import time
# import os
# import threading
# import queue
# import fpdf
# import datetime

# import logging

# import torch
# import torch.serialization

# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# from flask_cors import CORS
# CORS(app)

# # Global variables
# camera = None
# detection_thread = None
# is_running = False
# results_queue = queue.Queue()
# detected_words = []
# output_file = "detected_signs.txt"
# stop_event = threading.Event()
# speech_active = False
# speech_stop_event = threading.Event()
# tts_enabled = False  # Initialize TTS toggle state
# tts_lock = threading.Lock()  # Lock for thread-safe TTS operations
# tts_queue = queue.Queue()  # Queue for TTS requests

# # Initialize TTS engine
# import pyttsx3
# tts_engine = None

# def initialize_tts_engine():
#     """Initialize the TTS engine in its own thread"""
#     global tts_engine
#     try:
#         tts_engine = pyttsx3.init()
#         tts_engine.setProperty('rate', 150)
#         tts_engine.setProperty('volume', 0.9)
#         return True
#     except Exception as e:
#         app.logger.error(f"Error initializing TTS engine: {e}")
#         return False

# # Initialize TTS engine at startup
# initialize_tts_engine()

# # Start TTS worker thread
# def tts_worker():
#     """Worker thread to handle TTS requests"""
#     global tts_queue, tts_engine, speech_stop_event
    
#     while True:
#         try:
#             text = tts_queue.get(timeout=1.0)
#             if text == "STOP":
#                 # Special message to stop current speech
#                 with tts_lock:
#                     if tts_engine:
#                         tts_engine.stop()
#                 continue
                
#             if not speech_stop_event.is_set() and tts_engine:
#                 # Speak the text
#                 with tts_lock:
#                     tts_engine.say(text)
#                     tts_engine.runAndWait()
            
#             tts_queue.task_done()
#         except queue.Empty:
#             continue
#         except Exception as e:
#             app.logger.error(f"TTS worker error: {e}")
#             time.sleep(0.5)

# # Start TTS worker thread
# tts_thread = threading.Thread(target=tts_worker, daemon=True)
# tts_thread.start()

# def speak_text(text):
#     """Add text to the TTS queue"""
#     global tts_queue, tts_enabled
    
#     if not text or not tts_enabled:
#         return
        
#     try:
#         # Add to queue for TTS worker
#         tts_queue.put(text)
#     except Exception as e:
#         app.logger.error(f"Error queuing TTS: {e}")

# def stop_speaking():
#     """Stop all TTS output"""
#     global tts_queue, speech_stop_event
    
#     try:
#         speech_stop_event.set()
#         tts_queue.put("STOP")  # Special message to stop current speech
#         time.sleep(0.1)
#         speech_stop_event.clear()
#     except Exception as e:
#         app.logger.error(f"Error stopping TTS: {e}")

# def load_model():
#     # Load YOLO model from Ultralytics
#     try:
#         model_path = os.path.join(os.getcwd(), "best.pt")
#         if not os.path.exists(model_path):
#             # If the model isn't in the current directory, try a default path
#             model_path = "best.pt"
        
#         # Load the model using YOLO class
#         model = YOLO(model_path)
#         return model
#     except Exception as e:
#         app.logger.error(f"Error loading model: {e}")
#         raise RuntimeError(f"Failed to load model: {e}")

# def initialize_camera():
#     print("camera opened")
#     # Initialize camera
#     cam = cv2.VideoCapture(0)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
#     if not cam.isOpened():
#         raise RuntimeError("Could not access the camera.")
#     return cam

# def detection_loop(model):
#     global camera, is_running, detected_words, stop_event, tts_enabled
    
#     camera = initialize_camera()
    
#     # Track the last detected gesture
#     last_detected = set()
    
#     # Open file to save detections
#     with open(output_file, "w") as f:
#         f.write(f"Sign Language Detection Results - {datetime.datetime.now()}\n")
#         f.write("----------------------------------------\n")

#     # Cooldown timer for TTS to avoid constant repetition
#     last_speak_time = 0
#     speak_cooldown = 2.0  # seconds between TTS announcements
    
#     while not stop_event.is_set():
#         success, frame = camera.read()
#         if not success:
#             break
        
#         # Run detection on the frame
#         results = model(frame)
        
#         # Process results
#         current_detected = set()
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Get the class name
#                 cls = int(box.cls[0])
#                 class_name = model.names[cls]
#                 conf = float(box.conf[0])
                
#                 if conf > 0.3:  # Confidence threshold
#                     # Add to current frame detections
#                     current_detected.add(class_name)
                    
#                     # Draw bounding box on frame
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{class_name} {conf:.2f}", 
#                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         # Only record if there's a change in detected gestures
#         current_time = time.time()
#         if current_detected and current_detected != last_detected:
#             detection_str = ", ".join(current_detected)
#             timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
#             # Log to file
#             with open(output_file, "a") as f:
#                 f.write(f"[{timestamp}] Detected: {detection_str}\n")
            
#             # Update detected words list
#             detected_words.extend(current_detected)
            
#             # Speak detected signs if TTS is enabled and cooldown elapsed
#             if tts_enabled and (current_time - last_speak_time) > speak_cooldown:
#                 print(f"Speaking: {detection_str}")  # Debug log
#                 speak_text(detection_str)
#                 last_speak_time = current_time
            
#             # Update last detected
#             last_detected = current_detected.copy()
        
#         # Convert frame to JPEG for streaming
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
        
#         # Put the frame in the queue for the generator
#         results_queue.put(frame_bytes)
        
#         # Short sleep to reduce CPU usage
#         time.sleep(0.01)
    
#     # Release camera when done
#     if camera is not None:
#         camera.release()

# def generate_pdf():
#     """Generate a PDF file with the detected signs"""
#     pdf = fpdf.FPDF()
#     pdf.add_page()
    
#     # Set font
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Sign Language Detection Results", ln=True, align="C")
#     pdf.ln(10)
    
#     # Add timestamp
#     pdf.set_font("Arial", "", 12)
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     pdf.cell(0, 10, f"Generated on: {now}", ln=True)
#     pdf.ln(10)
    
#     # Add detected words
#     pdf.set_font("Arial", "B", 14)
#     pdf.cell(0, 10, "Detected Signs:", ln=True)
#     pdf.ln(5)
    
#     pdf.set_font("Arial", "", 12)
    
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             for line in f:
#                 pdf.cell(0, 10, line.strip(), ln=True)
#     else:
#         pdf.cell(0, 10, "No detections recorded", ln=True)
    
#     pdf_path = "sign_language_detection_results.pdf"
#     pdf.output(pdf_path)
#     return pdf_path

# @app.route('/')
# def index():
#     """Return the main page."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in your HTML src."""
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def generate_frames():
#     """Generator function for video streaming."""
#     global results_queue
    
#     while True:
#         try:
#             # Get frame from the queue with a timeout
#             frame_bytes = results_queue.get(timeout=1.0)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
#         except queue.Empty:
#             # If no new frame is available, send a placeholder or wait
#             continue
#         except Exception as e:
#             app.logger.error(f"Error in generate_frames: {e}")
#             break

# @app.route('/download_pdf', methods=['GET'])
# def download_pdf():
#     """Generate and download the PDF with detection results."""
#     try:
#         pdf_path = generate_pdf()
#         return send_file(pdf_path, as_attachment=True)
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)})

# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     global detection_thread, is_running, detected_words, stop_event, results_queue
#     try:
#         print("Start detection triggered!")  # Debug log
        
#         if not is_running:
#             # Clear previous results
#             detected_words = []
#             stop_event.clear()
            
#             # Clear any existing items in the queue
#             while not results_queue.empty():
#                 results_queue.get_nowait()
            
#             # Create and start detection thread
#             model = load_model()
#             detection_thread = threading.Thread(target=detection_loop, args=(model,))
#             detection_thread.daemon = True
#             detection_thread.start()
            
#             is_running = True
#             print("Detection started.")
#             return jsonify({"status": "success", "message": "Detection started"})
        
#         return jsonify({"status": "warning", "message": "Detection already running"})

#     except Exception as e:
#         app.logger.error(f"Error starting detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/stop_detection', methods=['POST'])
# def stop_detection():
#     global is_running, stop_event
#     try:
#         print("Stop detection triggered!")  # Debug log
        
#         if is_running:
#             stop_event.set()
#             stop_speaking()  # Stop any ongoing speech
#             is_running = False
#             print("Detection stopped.")
#             return jsonify({"status": "success", "message": "Detection stopped"})
        
#         return jsonify({"status": "warning", "message": "Detection not running"})
    
#     except Exception as e:
#         app.logger.error(f"Error stopping detection: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/toggle_tts', methods=['POST'])
# def toggle_tts():
#     global tts_enabled
#     try:
#         # Flip the toggle
#         tts_enabled = not tts_enabled
#         print(f"TTS Enabled: {tts_enabled}")  # Debug log
        
#         # If turning off, stop any current speech
#         if not tts_enabled:
#             stop_speaking()

#         # Send updated status back to frontend
#         return jsonify({
#             "status": "success",
#             "tts_enabled": tts_enabled
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/test_tts', methods=['POST'])
# def test_tts():
#     """Test endpoint for TTS functionality"""
#     try:
#         data = request.get_json()
#         test_text = data.get('text', 'This is a test of the text to speech system')
        
#         # Directly use the speak_text function
#         speak_text(test_text)
        
#         return jsonify({
#             "status": "success",
#             "message": f"TTS test initiated with text: {test_text}"
#         })
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/speak_pdf', methods=['POST'])
# def speak_pdf():
#     global speech_active, speech_stop_event
    
#     try:
#         # Stop any current speech first
#         stop_speaking()
        
#         # Get text to speak
#         text = ""
#         if os.path.exists(output_file):
#             with open(output_file, 'r') as f:
#                 text = f.read()
#         else:
#             text = "No signs detected to read out."
        
#         # Set speech as active
#         speech_active = True
        
#         # Return text for browser-based speech
#         return jsonify({
#             "status": "success", 
#             "text": text,
#             "speech_active": True
#         })
#     except Exception as e:
#         app.logger.error(f"speak_pdf error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/stop_speech', methods=['POST'])
# def stop_speech():
#     global speech_active
    
#     try:
#         stop_speaking()
#         speech_active = False
#         return jsonify({"status": "success", "message": "Speech stopped"})
#     except Exception as e:
#         app.logger.error(f"stop_speech error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

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

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, Response, send_file, request, jsonify
from ultralytics import YOLO
import cv2
import time
import os
import threading
import queue
import fpdf
import datetime

import logging

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
    print("camera opened")
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    if not cam.isOpened():
        raise RuntimeError("Could not access the camera.")
    return cam

def detection_loop(model):
    global camera, is_running, detected_words, stop_event, tts_enabled, latest_detection, detection_time
    
    camera = initialize_camera()
    
    # Detection parameters with emphasis on ensuring single detection
    CONFIDENCE_THRESHOLD = 0.5  # Base confidence threshold
    MIN_DETECTION_FRAMES = 3    # Frames required for stable detection
    DETECTION_COOLDOWN = 0.7    # Time between detections
    CERTAINTY_FACTOR = 0.15     # Required confidence margin for a detection to override others
    
    # Detection tracking variables
    detection_counts = {}       # Track consecutive detections
    last_detection_time = 0
    last_detected = None
    
    # Open file to save detections
    with open(output_file, "w") as f:
        f.write(f"Sign Language Detection Results - {datetime.datetime.now()}\n")
        f.write("----------------------------------------\n")

    frame_count = 0
    
    while not stop_event.is_set():
        success, frame = camera.read()
        if not success:
            break
            
        frame_count += 1
        # Process every other frame
        if frame_count % 2 != 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            results_queue.put(buffer.tobytes())
            continue
        
        # Run detection on the frame
        results = model(frame)
        
        # Process results - collect all detections first
        all_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                
                if conf > CONFIDENCE_THRESHOLD:
                    all_detections.append({
                        'class_name': class_name,
                        'confidence': conf,
                        'box': box
                    })
        
        # Sort all detections by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If we have multiple high-confidence detections, 
        # only accept the one with significantly higher confidence
        highest_conf_detection = None
        current_frame_detections = {}
        
        if all_detections:
            highest_conf_detection = all_detections[0]
            highest_conf = highest_conf_detection['confidence']
            
            # Only accept this detection if it's significantly more confident than others
            accept_highest = True
            
            if len(all_detections) > 1:
                second_highest = all_detections[1]['confidence']
                # If there's a clear winner (significantly higher confidence)
                if highest_conf - second_highest < CERTAINTY_FACTOR:
                    accept_highest = False
            
            if accept_highest:
                class_name = highest_conf_detection['class_name']
                current_frame_detections[class_name] = highest_conf
                
                # Draw bounding box with improved visibility
                box = highest_conf_detection['box']
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add background to text for visibility
                text = f"{class_name} {highest_conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Update detection counts with decay
        for class_name in list(detection_counts.keys()):
            if class_name not in current_frame_detections:
                detection_counts[class_name] = max(0, detection_counts[class_name] - 0.7)
                if detection_counts[class_name] <= 0:
                    del detection_counts[class_name]
            else:
                # Weight by confidence
                detection_counts[class_name] += current_frame_detections[class_name]
        
        # Add new detections to counts
        for class_name in current_frame_detections:
            if class_name not in detection_counts:
                detection_counts[class_name] = 1.0
        
        # Check for stable detections
        current_time = time.time()
        stable_detections = [cls for cls, count in detection_counts.items() 
                            if count >= MIN_DETECTION_FRAMES]
        
        # Only report a detection if we have a single stable detection
        # and enough time has passed since the last detection
        if stable_detections and (current_time - last_detection_time) > DETECTION_COOLDOWN:
            # Sort by detection count and only take the most stable one
            stable_detections.sort(key=lambda cls: detection_counts[cls], reverse=True)
            # STRICTLY take only one detection
            stable_detections = stable_detections[:1]
            
            detection_str = stable_detections[0]
            
            # Only record if different from last recorded detection
            if detection_str != last_detected:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                with open(output_file, "a") as f:
                    f.write(f"[{timestamp}] Detected: {detection_str}\n")

                detected_words.append(detection_str)
                
                # Update latest detection for TTS
                if tts_enabled:
                    latest_detection = detection_str
                    detection_time = current_time
                    print(f"New detection: {detection_str}")
                
                # Partially reset counter for the detected class (cooldown period)
                detection_counts[detection_str] = MIN_DETECTION_FRAMES * 0.5
                
                # Update tracking variables
                last_detection_time = current_time
                last_detected = detection_str
        
        # Add visual confidence indicator
        cv2.putText(frame, "Detection Confidence", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        y_offset = 50
        # Show at most the top 3 candidates
        top_candidates = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for class_name, count in top_candidates:
            progress = min(1.0, count / MIN_DETECTION_FRAMES)
            bar_width = int(200 * progress)
            
            # Red if not enough confidence, green if sufficient
            bar_color = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)
            
            cv2.rectangle(frame, (20, y_offset), (20 + bar_width, y_offset + 15), 
                         bar_color, -1)
            cv2.rectangle(frame, (20, y_offset), (20 + 200, y_offset + 15), 
                         (255, 255, 255), 1)
            cv2.putText(frame, f"{class_name} ({count:.1f}/{MIN_DETECTION_FRAMES})", (230, y_offset + 13), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Show detection status
        status_text = "Ready for detection"
        if stable_detections:
            status_text = f"Detected: {stable_detections[0]}"
        elif highest_conf_detection:
            status_text = f"Considering: {highest_conf_detection['class_name']}"
            
        cv2.putText(frame, status_text, (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Put the frame in the queue for the generator
        results_queue.put(frame_bytes)
        
        # Keep responsive
        time.sleep(0.015)
    
    # Release camera when done
    if camera is not None:
        camera.release()

        
def generate_pdf():
    """Generate a PDF file with the detected signs"""
    pdf = fpdf.FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sign Language Detection Results", ln=True, align="C")
    pdf.ln(10)
    
    # Add timestamp
    pdf.set_font("Arial", "", 12)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated on: {now}", ln=True)
    pdf.ln(10)
    
    # Add detected words
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detected Signs:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                pdf.cell(0, 10, line.strip(), ln=True)
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
    
    while True:
        try:
            # Get frame from the queue with a timeout
            frame_bytes = results_queue.get(timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except queue.Empty:
            # If no new frame is available, send a placeholder or wait
            continue
        except Exception as e:
            app.logger.error(f"Error in generate_frames: {e}")
            break

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
        print("Start detection triggered!")  # Debug log
        
        if not is_running:
            # Clear previous results
            detected_words = []
            stop_event.clear()
            
            # Clear any existing items in the queue
            while not results_queue.empty():
                results_queue.get_nowait()
            
            # Create and start detection thread
            model = load_model()
            detection_thread = threading.Thread(target=detection_loop, args=(model,))
            detection_thread.daemon = True
            detection_thread.start()
            
            is_running = True
            print("Detection started.")
            return jsonify({"status": "success", "message": "Detection started"})
        
        return jsonify({"status": "warning", "message": "Detection already running"})

    except Exception as e:
        app.logger.error(f"Error starting detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_running, stop_event
    try:
        print("Stop detection triggered!")  # Debug log
        
        if is_running:
            stop_event.set()
            is_running = False
            print("Detection stopped.")
            return jsonify({"status": "success", "message": "Detection stopped"})
        
        return jsonify({"status": "warning", "message": "Detection not running"})
    
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

@app.route('/get_detection_status', methods=['GET'])
def get_detection_status():
    global is_running, detected_words, speech_active, tts_enabled
    
    try:
        return jsonify({
            "is_running": is_running,
            "detected_count": len(detected_words) if detected_words else 0,
            "speech_active": speech_active,
            "tts_enabled": tts_enabled
        })
    except Exception as e:
        app.logger.error(f"Error fetching status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

