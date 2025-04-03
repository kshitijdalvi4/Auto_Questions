import cv2
import pytesseract
import numpy as np
from summarizer import Summarizer
from difflib import SequenceMatcher

# Load BERT Summarization Model
bert_model = Summarizer()

# Capture live video (0 = Webcam, or provide video file path)
video_source = 0  # Use GMeet, Zoom, or YouTube screen recording
cap = cv2.VideoCapture(video_source)

previous_text = ""
frame_skip = 30  # Process every 30 frames (reduce CPU load)

def preprocess_frame(frame):
    """ Convert frame to grayscale and apply thresholding for better OCR """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def text_similarity(text1, text2):
    """ Compute similarity between two texts to detect content change """
    return SequenceMatcher(None, text1, text2).ratio()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:  # Process every 30 frames
        processed_frame = preprocess_frame(frame)
        extracted_text = pytesseract.image_to_string(processed_frame)

        # Check if the text has significantly changed
        similarity = text_similarity(previous_text, extracted_text)
        if similarity < 0.7:  # Slide/content change detected
            print("\n\n🔹 **Slide Changed! New Content Detected** 🔹\n")
            
            # Use BERT to summarize extracted text
            summary = bert_model(extracted_text, num_sentences=3)  # Extract key sentences
            print(f"📌 **Summary:**\n{summary}")
            
            previous_text = extracted_text  # Update for next iteration

    # Show the video feed (optional)
    cv2.imshow("Live OCR Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
