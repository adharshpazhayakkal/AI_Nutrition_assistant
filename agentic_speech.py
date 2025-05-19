import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import logging
import threading
import queue
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, filename="app.log")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# Mock Agent class for demonstration (replace with actual agno.agent implementation)
class Agent:
    def __init__(self, model, description, instructions, tools=None, show_tool_calls=False, markdown=True):
        self.model = model
        self.description = description
        self.instructions = instructions
        self.tools = tools or []
        self.show_tool_calls = show_tool_calls
        self.markdown = markdown

    def run(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return type('Response', (), {'content': response.text})()
        except Exception as e:
            return type('Response', (), {'content': f"Error: {str(e)}"})()

# Mock Gemini model (replace with actual agno.models.google.Gemini)
class Gemini:
    def __init__(self, id):
        self.id = id
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_content(self, prompt):
        return self.model.generate_content(prompt)

# Mock DuckDuckGoTools (replace with actual agno.tools.duckduckgo.DuckDuckGoTools)
class DuckDuckGoTools:
    pass

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Speech-to-Text function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak clearly and then wait.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success("Transcription successful!")
            return text
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Could not understand the audio. Please speak clearly."
        except sr.RequestError as e:
            return f"Speech recognition error: {str(e)}"

# Text-to-Speech function
def text_to_speech(text, filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        with open(filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return audio_b64
    except Exception as e:
        return f"Error generating audio: {str(e)}"

# Timeout mechanism
def run_with_timeout(func, timeout, *args, **kwargs):
    result_queue = queue.Queue()
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return "Response generation timed out. Try a shorter query."
    result, value = result_queue.get()
    if result == "error":
        return f"Error generating response: {value}"
    return value

# Input sanitization
def sanitize_input(user_input):
    user_input = user_input.strip()
    if not user_input:
        raise ValueError("Input cannot be empty.")
    if len(user_input) > 200:
        raise ValueError("Input is too long. Keep it under 200 characters.")
    words = user_input.split()
    if len(words) > 5 and len(set(words)) < len(words) / 3:
        raise ValueError("Input contains excessive repetition. Provide a concise query.")
    return user_input

# Output cleaning
def clean_output(text):
    sentences = text.split(". ")
    unique_sentences = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            unique_sentences.append(s)
            seen.add(s)
    cleaned = ". ".join(unique_sentences)
    if len(cleaned) < 20 or any(word in cleaned.lower() for word in ["sugar", "banana", "milk"]) and "low-carb" in st.session_state.get("user_input", "").lower():
        return "Invalid response. Try again or refine the query for low-carb options."
    return cleaned

# Dietary Planner Agent
dietary_planner = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="Creates personalized dietary plans based on user input.",
    instructions=[
        "Generate a diet plan with breakfast, lunch, dinner, and snacks.",
        "Consider dietary preferences like Keto, Vegetarian, or Low Carb.",
        "Ensure proper hydration and electrolyte balance.",
        "Provide nutritional breakdown including macronutrients and vitamins.",
        "Suggest meal preparation tips for easy implementation.",
        "If necessary, search the web for additional information.",
    ],
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

def get_meal_plan(age, weight, height, activity_level, dietary_preference, fitness_goal):
    prompt = (f"Create a personalized meal plan for a {age}-year-old person, weighing {weight}kg, "
              f"{height}cm tall, with an activity level of '{activity_level}', following a "
              f"'{dietary_preference}' diet, aiming to achieve '{fitness_goal}'.")
    return dietary_planner.run(prompt)

# Fitness Trainer Agent
fitness_trainer = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="Generates customized workout routines based on fitness goals.",
    instructions=[
        "Create a workout plan including warm-ups, main exercises, and cool-downs.",
        "Adjust workouts based on fitness level: Beginner, Intermediate, Advanced.",
        "Consider weight loss, muscle gain, endurance, or flexibility goals.",
        "Provide safety tips and injury prevention advice.",
        "Suggest progress tracking methods for motivation.",
        "If necessary, search the web for additional information.",
    ],
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

def get_fitness_plan(age, weight, height, activity_level, fitness_goal):
    prompt = (f"Generate a workout plan for a {age}-year-old person, weighing {weight}kg, "
              f"{height}cm tall, with an activity level of '{activity_level}', "
              f"aiming to achieve '{fitness_goal}'. Include warm-ups, exercises, and cool-downs.")
    return fitness_trainer.run(prompt)

# Team Lead Agent
team_lead = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="Combines diet and workout plans into a holistic health strategy.",
    instructions=[
        "Merge personalized diet and fitness plans for a comprehensive approach, Use Tables if possible.",
        "Ensure alignment between diet and exercise for optimal results.",
        "Suggest lifestyle tips for motivation and consistency.",
        "Provide guidance on tracking progress and adjusting plans over time."
    ],
    markdown=True
)

def get_full_health_plan(name, age, weight, height, activity_level, dietary_preference, fitness_goal):
    meal_plan = get_meal_plan(age, weight, height, activity_level, dietary_preference, fitness_goal)
    fitness_plan = get_fitness_plan(age, weight, height, activity_level, fitness_goal)
    return team_lead.run(
        f"Greet the customer, {name}\n\n"
        f"User Information: {age} years old, {weight}kg, {height}cm, activity level: {activity_level}.\n\n"
        f"Fitness Goal: {fitness_goal}\n\n"
        f"Meal Plan:\n{meal_plan.content}\n\n"
        f"Workout Plan:\n{fitness_plan.content}\n\n"
        f"Provide a holistic health strategy integrating both plans."
    )

# Process text input for nutrition queries
def process_text_input(user_input, age, weight, condition, preferences, allergies):
    logging.info("Processing text input: %s", user_input[:50])
    prompt = f"""
    You are an expert in personalized nutrition. Provide accurate, concise, and culturally relevant meal plans or dietary advice tailored to the user’s health goals, medical conditions, fitness routines, allergies, and preferences. Use USDA FoodData Central or Indian Food Composition Tables for nutritional data. Avoid high-carb ingredients for low-carb requests. Provide:
    1. Meal suggestion (ingredients, preparation steps).
    2. Nutrition (calories, carbs, protein, fat per serving).
    3. Explanation (why the meal suits the user’s needs).
    4. Optional: Exercise suggestions if relevant.

    User Input: {user_input}
    User Details: Age: {age}, Weight: {weight}kg, Condition: {condition}, Preferences: {preferences}, Allergies: {allergies}

    Response Format:
    **Meal Suggestion**: [Meal name, ingredients, preparation]
    **Nutrition**: [Calories, carbs, protein, fat]
    **Explanation**: [Why this meal is suitable]
    **Exercise (if relevant)**: [Exercise, duration, calories burned]
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = clean_output(response.text)
        return cleaned_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Process image input for food analysis
def process_image_input(uploaded_file, user_input, age, weight, condition, preferences, allergies):
    logging.info("Processing image input: %s", user_input[:50])
    try:
        if uploaded_file is None:
            raise ValueError("Please upload a food image.")
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_data = [{"mime_type": f"image/{image.format.lower()}", "data": image_bytes.getvalue()}]
        
        prompt = f"""
        You are an expert in analyzing food images for nutritional details. Your goal is to:
        1. Identify food items and portion sizes in the image.
        2. Calculate total calories and macronutrients using USDA FoodData Central or Indian Food Composition Tables.
        3. Provide a health assessment (suitability for user’s condition).
        4. Suggest exercises to burn the meal’s calories.
        5. Offer smart food swaps if the meal is unhealthy.

        User Input: {user_input}
        User Details: Age: {age}, Weight: {weight}kg, Condition: {condition}, Preferences: {preferences}, Allergies: {allergies}

        Response Format:
        **Food Identified**: [Listascape**: [List of items, portion sizes]
        **Total Nutrition**: [Calories, carbs, protein, fat]
        **Itemized Breakdown**: [Nutrition per item]
        **Health Assessment**: [Suitability for user’s condition]
        **Exercise Suggestions**: [Exercise, duration, calories burned]
        **Smart Swaps (if needed)**: [Healthier alternatives]
        """
        response = model.generate_content([prompt] + image_data)
        cleaned_response = clean_output(response.text)
        return cleaned_response
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Health & Nutrition Assistant", page_icon="🏋️‍♂️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .title { text-align: center; font-size: 48px; font-weight: bold; color: #FF6347; }
        .subtitle { text-align: center; font-size: 24px; color: #4CAF50; }
        .sidebar { background-color: #F5F5F5; padding: 20px; border-radius: 10px; }
        .content { padding: 20px; background-color: #E0F7FA; border-radius: 10px; margin-top: 20px; }
        .btn { background-color: #FF6347; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; text-decoration: none; }
        .goal-card { padding: 20px; margin: 10px; background-color: #FFF; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<h1 class="title">🏋️‍♂️ AI Health & Nutrition Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Personalized fitness, nutrition plans, and food analysis with voice input!</p>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("⚙️ User Details")
name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.number_input("Age (years)", min_value=10, max_value=100, value=30)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
activity_level = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
dietary_preference = st.sidebar.selectbox("Dietary Preference", ["Keto", "Vegetarian", "Low Carb", "Balanced"])
fitness_goal = st.sidebar.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"])
condition = st.sidebar.selectbox("Medical Condition", ["None", "Type 2 Diabetes", "Hypertension", "PCOS", "IBS", "Heart Disease"])
preferences = st.sidebar.text_input("Preferences (e.g., vegetarian, Kerala cuisine)", "")
allergies = st.sidebar.text_input("Allergies (e.g., dairy, nuts)", "")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Health & Fitness Plan", "Nutrition Query", "Food Image Analysis"])

# Tab 1: Health & Fitness Plan
with tab1:
    st.header("Personalized Health & Fitness Plan")
    if st.button("Generate Health Plan"):
        if not (age and weight and height and name.strip()):
            st.warning("Please fill in all required fields.")
        else:
            with st.spinner("Generating your health & fitness plan..."):
                full_health_plan = get_full_health_plan(name, age, weight, height, activity_level, dietary_preference, fitness_goal)
                st.subheader(f"{name}'s Health Strategy")
                st.markdown(full_health_plan.content)
                st.info("This is your customized health and fitness strategy.")
                st.markdown("""
                    <div class="goal-card">
                        <h4>🏆 Stay Focused, Stay Fit!</h4>
                        <p>Consistency is key! Keep pushing, and you will see results!</p>
                    </div>
                """, unsafe_allow_html=True)

# Tab 2: Nutrition Query
with tab2:
    st.header("Nutrition Query")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Enter your query (e.g., 'Suggest a low-carb breakfast')")
    with col2:
        if st.button("🎙️ Record Query"):
            user_input = speech_to_text()
            st.text_area("Transcribed Query", value=user_input, height=100)
    if st.button("Submit Query"):
        if not user_input.strip():
            st.warning("Please enter or record a query.")
        else:
            with st.spinner("Generating response..."):
                try:
                    user_input = sanitize_input(user_input)
                    st.session_state["user_input"] = user_input
                    text_response = run_with_timeout(
                        process_text_input, 20, user_input, age, weight, condition, preferences, allergies
                    )
                    if "Error" in text_response or "timed out" in text_response:
                        st.error(text_response)
                    else:
                        st.subheader("Response")
                        st.markdown(text_response)
                        # Text-to-Speech
                        audio_b64 = text_to_speech(text_response)
                        if not audio_b64.startswith("Error"):
                            st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                        else:
                            st.error(audio_b64)
                        response_data = {
                            "input": user_input,
                            "age": age,
                            "weight": weight,
                            "condition": condition,
                            "preferences": preferences,
                            "allergies": allergies,
                            "response": text_response
                        }
                        with open("text_response.json", "w") as f:
                            json.dump(response_data, f, indent=2)
                        with open("text_response.json", "rb") as f:
                            st.download_button(
                                label="Download Text Response",
                                data=f,
                                file_name="text_response.json",
                                mime="application/json"
                            )
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error processing text input: {str(e)}")

# Tab 3: Food Image Analysis
with tab3:
    st.header("Food Image Analysis")
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns([3, 1])
    with col1:
        image_input = st.text_input("Describe the meal (e.g., 'South Indian meal')")
    with col2:
        if st.button("🎙️ Record Description"):
            image_input = speech_to_text()
            st.text_area("Transcribed Description", value=image_input, height=100)
    if st.button("Analyze Image"):
        if not (uploaded_file or image_input.strip()):
            st.warning("Please upload an image or provide a description.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    image_input = sanitize_input(image_input) if image_input.strip() else "Analyze the meal"
                    image_response = run_with_timeout(
                        process_image_input, 20, uploaded_file, image_input, age, weight, condition, preferences, allergies
                    )
                    if "Error" in image_response or "timed out" in image_response:
                        st.error(image_response)
                    else:
                        st.subheader("Image Analysis")
                        st.markdown(image_response)
                        # Text-to-Speech
                        audio_b64 = text_to_speech(image_response)
                        if not audio_b64.startswith("Error"):
                            st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")
                        else:
                            st.error(audio_b64)
                        response_data = {
                            "input": image_input,
                            "age": age,
                            "weight": weight,
                            "condition": condition,
                            "preferences": preferences,
                            "allergies": allergies,
                            "response": image_response
                        }
                        with open("image_response.json", "w") as f:
                            json.dump(response_data, f, indent=2)
                        with open("image_response.json", "rb") as f:
                            st.download_button(
                                label="Download Image Analysis",
                                data=f,
                                file_name="image_response.json",
                                mime="application/json"
                            )
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error processing image input: {str(e)}")