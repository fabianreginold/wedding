from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import logging
import json
import os
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from functools import wraps

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wedding_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Google Sheets setup
def get_google_sheet():
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds_file = 'credentials.json'
        
        if not os.path.exists(creds_file):
            raise FileNotFoundError(f"Credentials file '{creds_file}' not found")

        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
        client = gspread.authorize(creds)
        spreadsheet_key = '1ddmRlECMSSXMKnlG5T_SrO578DRZ5uONvUZALrTtj-8'
        sheet = client.open_by_key(spreadsheet_key).worksheet('rsvp')
        return sheet
    except Exception as e:
        logger.error(f"Error initializing Google Sheets: {str(e)}", exc_info=True)
        raise

# API Configuration
GOOGLE_API_KEY = "AIzaSyCJLUqVQt8g2TKPS_ztTAMlWONuZ-AKSuI"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Enhanced Wedding FAQ Data
FAQ_FILE = "faqs.json"
WEDDING_FAQ = []
if os.path.exists(FAQ_FILE):
    try:
        with open(FAQ_FILE, 'r') as f:
            WEDDING_FAQ = json.load(f)
        logger.info(f"Loaded {len(WEDDING_FAQ)} FAQ entries from {FAQ_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{FAQ_FILE}': {e}")
else:
    logger.error(f"FAQ file '{FAQ_FILE}' not found. Using an empty list.")

# Enhanced stop words list
STOP_WORDS = set(["a", "an", "the", "is", "of", "and", "in", "for", "with", "at", "by", "from", "on", "to", 
                 "what", "where", "when", "how", "are", "do", "does", "will", "i", "me", "my", "you", "your", 
                 "it", "its", "we", "our", "us", "be", "was", "were", "have", "has", "had", "can", "could", "should"])

# Decorator for logging API calls
def log_api_call(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"API Call: {request.method} {request.path}")
        logger.debug(f"Headers: {dict(request.headers)}")
        if request.method in ['POST', 'PUT']:
            logger.debug(f"Request data: {request.get_json(silent=True) or request.form}")
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/api")
@log_api_call
def api_info():
    return jsonify({
        "status": "success",
        "message": "Welcome to Fabian & Sarah's Wedding API",
        "endpoints": {
            "/api/submit-rsvp": "POST - Submit RSVP data",
            "/api/chat": "POST - Chat with wedding assistant",
            "/chat": "POST - Chat with wedding assistant (legacy endpoint)"
        }
    })

@app.route("/api/submit-rsvp", methods=["POST"])
@log_api_call
def submit_rsvp():
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        if not data:
            logger.error("No data received")
            return jsonify({
                "status": "error",
                "message": "No data received",
                "received_data": str(data)
            }), 400

        logger.info(f"Received RSVP data: {json.dumps(data, indent=2)}")

        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'address', 'attending', 'guestCount']
        missing_fields = [field for field in required_fields if field not in data or not str(data[field]).strip()]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "missing_fields": missing_fields
            }), 400

        # Convert guestCount to string if it's not already
        guest_count_str = str(data['guestCount']).strip()
        
        # Additional validation
        if not guest_count_str.isdigit():
            logger.error(f"Invalid guest count: {data['guestCount']}")
            return jsonify({
                "status": "error",
                "message": "Guest count must be a number",
                "invalid_field": "guestCount"
            }), 400

        guest_count = int(guest_count_str)
        additional_guests = data.get('additionalGuests', [])
        
        # Handle case where additionalGuests might be a string (from form data)
        if isinstance(additional_guests, str):
            try:
                additional_guests = json.loads(additional_guests) if additional_guests else []
            except json.JSONDecodeError:
                additional_guests = []

        if len(additional_guests) != guest_count - 1:
            logger.warning(f"Guest count mismatch: expected {guest_count-1} additional guests, got {len(additional_guests)}")
            # Don't fail the request, just log the warning

        sheet = get_google_sheet()
        
        # Prepare rows to add
        rows_to_add = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Main guest row
        main_guest_row = [
            str(data['firstName']).strip(),
            str(data['lastName']).strip(),
            str(data['email']).strip(),
            str(data['address']).strip(),
            str(data['attending']).strip(),
            guest_count_str,
            str(data.get('dietary', '')).strip(),
            timestamp
        ]
        rows_to_add.append(main_guest_row)
        
        # Additional guests rows
        for guest in additional_guests:
            if not isinstance(guest, dict):
                logger.warning(f"Invalid guest data format: {guest}")
                continue
                
            if not guest.get('firstName') or not guest.get('lastName'):
                logger.warning(f"Incomplete guest data: {guest}")
                continue
                
            guest_row = [
                str(guest['firstName']).strip(),
                str(guest['lastName']).strip(),
                str(data['email']).strip(),  # Same email as main guest
                str(data['address']).strip(),  # Same address as main guest
                str(data['attending']).strip(),  # Same attendance status
                "",  # Empty for guest count (only main row has this)
                str(data.get('dietary', '')).strip(),  # Same dietary restrictions
                timestamp
            ]
            rows_to_add.append(guest_row)
        
        # Add all rows in batch
        try:
            sheet.append_rows(rows_to_add)
            logger.info(f"Successfully added {len(rows_to_add)} RSVP rows to Google Sheet")
            
            response_data = {
                "status": "success",
                "message": "Thank you for your RSVP! We have received your response.",
                "data": {
                    "main_guest": f"{data['firstName']} {data['lastName']}",
                    "guest_count": guest_count_str,
                    "timestamp": timestamp,
                    "rows_added": len(rows_to_add)
                }
            }
            
            logger.info(f"Sending success response: {response_data}")
            return jsonify(response_data), 200

        except Exception as e:
            logger.error(f"Google Sheets API error: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "message": "There was an error saving your RSVP. Please try again later.",
                "error_details": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in submit_rsvp: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred. Please try again or contact us directly.",
            "error_details": str(e)
        }), 500

@app.route("/api/chat", methods=["POST"])
@log_api_call
def api_chat():
    return chat_handler()

@app.route("/chat", methods=["POST"])
@log_api_call
def chat():
    return chat_handler()

def chat_handler():
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "Please provide a message"
            }), 400

        user_question = str(data['message']).strip()
        if not user_question:
            return jsonify({
                "status": "error",
                "message": "Please enter a question about Fabian and Sarah's wedding."
            }), 400

        logger.info(f"Chat request: {user_question}")
        
        # Enhanced FAQ matching with context extraction
        context = extract_relevant_context(user_question.lower())
        
        # Always use Gemini with the extracted context for more natural conversations
        llm_response = ask_gemini(user_question, context=context)
        if llm_response:
            return jsonify({
                "status": "success",
                "response": llm_response
            })

        # Fallback response if Gemini fails
        fallback_msg = "I'd love to share more about Fabian and Sarah! They met through their church in December 2024. Sarah first noticed Fabian's warm and genuine nature, while Fabian was immediately drawn to Sarah's beautiful smile. What would you like to know more about?"
        return jsonify({
            "status": "success",
            "response": fallback_msg
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "I'm having a little trouble answering right now. Could you ask me something else about Fabian and Sarah's wedding?"
        }), 500

def extract_relevant_context(question: str) -> str:
    """Extract relevant context from FAQs based on the question."""
    if not WEDDING_FAQ:
        return ""
    
    # First try exact matches
    for faq in WEDDING_FAQ:
        if faq.get("question", "").lower() in question:
            return faq.get("answer", "")
    
    # Then try keyword matching
    user_words = set(re.findall(r'\b\w+\b', question))
    relevant_user_keywords = user_words.difference(STOP_WORDS)
    
    best_context = ""
    best_score = 0
    
    for faq in WEDDING_FAQ:
        faq_question = faq.get("question", "").lower()
        faq_words = set(re.findall(r'\b\w+\b', faq_question))
        relevant_faq_keywords = faq_words.difference(STOP_WORDS)
        
        intersection = relevant_user_keywords.intersection(relevant_faq_keywords)
        union = relevant_user_keywords.union(relevant_faq_keywords)
        
        if len(union) > 0:
            score = len(intersection) / len(union)
            if score > best_score:
                best_score = score
                best_context = faq.get("answer", "")
    
    return best_context

def ask_gemini(question: str, context: str = None) -> str | None:
    """Enhanced Gemini prompt with more personality and context handling."""
    try:
        logger.info(f"Sending question to Gemini: {question}")
        headers = {"Content-Type": "application/json"}
        
        # Enhanced prompt with more personality and guidance
        prompt = f"""You are the friendly wedding assistant for Fabian and Sarah's wedding. Your name is Joy. 
        Your role is to provide warm, personal, and engaging answers about the couple and their wedding.

        Personality traits:
        - Warm and enthusiastic
        - Knowledgeable about Fabian and Sarah
        - Conversational but professional
        - Helpful and patient
        - Adds personal touches when appropriate

        About Fabian and Sarah:
        - They met through their church in December 2024
        - Sarah's first impression of Fabian was his warm and genuine nature
        - Fabian first noticed Sarah's beautiful smile
        - They got engaged in January 2025 after introducing each other to their families
        - Wedding date: March 7, 2026
        - Venue: New Testament Church, Basking Ridge, NJ

        {f"Additional context that might help: {context}" if context else ""}

        Guidelines:
        1. Always respond in a warm, conversational tone like you're talking to a friend
        2. Share interesting details about Fabian and Sarah when relevant
        3. If you don't know something, politely say so and suggest asking the couple
        4. Keep responses concise but informative (2-3 sentences typically)
        5. For personal questions, focus on what's appropriate to share publicly
        6. Add emojis occasionally to make it friendlier (but don't overdo it)

        Current question: {question}

        Please respond naturally as if you're having a friendly conversation:"""

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.8,  # Slightly higher for more creative responses
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        logger.warning("Gemini response missing 'candidates'")
        return None

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error from Gemini: {http_err} - {response.text if 'response' in locals() else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "error": str(error)
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "error": str(error)
    }), 500

# Add OPTIONS method handler for CORS preflight
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
