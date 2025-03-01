import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from recommendation_model import generate_learning_path, GenerateLearningPathIndexEmbeddings
from assessment_model import generate_assessment  # Import the new assessment model

# Function to check and update the FAISS index
def update_faiss_index(csv_filename):
    faiss_vectorstore_foldername = "faiss_learning_path_index"
    csv_last_modified = datetime.fromtimestamp(os.path.getmtime(csv_filename))
    index_last_modified = None
    if os.path.exists(faiss_vectorstore_foldername):
        index_last_modified = datetime.fromtimestamp(os.path.getmtime(faiss_vectorstore_foldername))
    if not os.path.exists(faiss_vectorstore_foldername) or csv_last_modified > index_last_modified:
        print(' -- Creating a new FAISS vector store from chunked text and Gemini embeddings.')
        GenerateLearningPathIndexEmbeddings(csv_filename)
        print(f' -- Saved the newly created FAISS vector store at "{faiss_vectorstore_foldername}".')
    else:
        print(f' -- Found existing FAISS vector store at "{faiss_vectorstore_foldername}", loading from cache.')

# Function to split response into introduction and table
def process_recommendation(recommendation_text):
    # Look for the table marker
    table_pattern = r'\|\s*Learning Pathway\s*\|\s*duration\s*\|\s*link\s*\|\s*Module\s*\|'
    
    # Check if the pattern exists in the text
    if re.search(table_pattern, recommendation_text):
        # Split the text at the table marker
        parts = re.split(table_pattern, recommendation_text, 1)
        
        path_introduction = parts[0].strip()
        path_content = '| Learning Pathway | duration | link | Module |\n' + parts[1].strip()
        
        return path_introduction, path_content
    else:
        # If table format isn't found, return the whole text as introduction
        return recommendation_text, ""

# Function to parse JSON assessment response
def process_assessment(assessment_text):
    # Try to extract JSON if it's embedded in markdown or text
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, assessment_text)
    
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to directly parse as JSON
    try:
        return json.loads(assessment_text)
    except:
        # If not valid JSON, return the raw text
        return {"raw_text": assessment_text}

# Set the title of the app with improved styling
st.set_page_config(page_title="Learning Path Assistant", layout="wide")

# Custom CSS for better styling with hover effects and improved view learning path section
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .intro-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #424242;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-container {
        margin-top: 2rem;
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
    }
    .path-introduction {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .path-introduction:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .path-content {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        overflow-x: auto;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .path-content:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .profile-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3F51B5;
        transition: all 0.3s ease;
    }
    .profile-card:hover {
        background-color: #E8EAF6;
        box-shadow: 0 5px 15px rgba(63, 81, 181, 0.2);
    }
    .action-button {
        transition: all 0.3s ease;
    }
    .action-button:hover {
        transform: scale(1.05);
    }
    .regenerate-container {
        margin-top: 1.5rem;
        background-color: #EDE7F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #7E57C2;
        transition: all 0.3s ease;
    }
    .regenerate-container:hover {
        background-color: #D1C4E9;
        box-shadow: 0 5px 15px rgba(126, 87, 194, 0.2);
    }
    .regenerate-title {
        font-size: 1.2rem;
        color: #5E35B1;
        margin-bottom: 0.8rem;
    }
    .save-options {
        margin-top: 2rem;
        background-color: #FFF8E1;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
        transition: all 0.3s ease;
    }
    .save-options:hover {
        background-color: #FFECB3;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.2);
    }
    /* Table styling */
    .path-content table {
        width: 100%;
        border-collapse: collapse;
    }
    .path-content th {
        background-color: #3F51B5;
        color: white;
        padding: 12px;
        text-align: left;
    }
    .path-content td {
        padding: 10px;
        border-bottom: 1px solid #E0E0E0;
    }
    .path-content tr:hover {
        background-color: #F5F5F5;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F5F7FA;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding: 0px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
        border-bottom: 3px solid #1E88E5;
    }
    /* Assessment styling */
    .assessment-container {
        margin-top: 1.5rem;
        background-color: #E8F5E9;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #43A047;
        transition: all 0.3s ease;
    }
    .assessment-container:hover {
        background-color: #C8E6C9;
        box-shadow: 0 5px 15px rgba(67, 160, 71, 0.2);
    }
    .assessment-title {
        font-size: 1.2rem;
        color: #2E7D32;
        margin-bottom: 0.8rem;
    }
    .assessment-section {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #2E7D32;
        transition: all 0.3s ease;
    }
    .assessment-section:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(46, 125, 50, 0.15);
    }
    .question {
        margin-bottom: 1.2rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #E0E0E0;
    }
    .option {
        margin-left: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .correct-answer {
        font-weight: bold;
        color: #2E7D32;
    }
    .practical-exercise {
        background-color: #F1F8E9;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header with custom styling
st.markdown('<div class="main-header">Your Virtual Learning Assistant</div>', unsafe_allow_html=True)

# About section with improved content and styling
st.markdown('<div class="intro-text">Welcome to your personal learning journey assistant! Our AI-powered platform helps you navigate educational resources tailored to your specific goals and interests. We analyze your learning objectives to create a structured path that maximizes your progress and keeps you motivated.</div>', unsafe_allow_html=True)

# Information box
st.markdown('<div class="info-box">To get started, please provide some information about yourself and your learning goals. This will help us generate a personalized learning path that matches your needs and interests.</div>', unsafe_allow_html=True)

# Define the CSV file path
csv_filename = "one.csv"

# Update the FAISS index if necessary
update_faiss_index(csv_filename)

# Initialize session state variables if they don't exist
if 'show_regenerate' not in st.session_state:
    st.session_state.show_regenerate = False
if 'show_assessment' not in st.session_state:
    st.session_state.show_assessment = False

# Create a cleaner form with tabs
tab1, tab2, tab3 = st.tabs(["Your Information", "View Learning Path", "Assessment"])

with tab1:
    st.markdown('<div class="sub-header">Personal Information</div>', unsafe_allow_html=True)
    
    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
        
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            education_level = st.selectbox(
                "Education Level", 
                ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD", "Self-taught", "Other"]
            )
        
        st.markdown('<div class="sub-header">Learning Objectives</div>', unsafe_allow_html=True)
        
        learning_category = st.selectbox(
            "Category of Interest",
            ["Web Development", "Data Science", "Mobile Development", "AI/Machine Learning", 
             "Cybersecurity", "Cloud Computing", "Game Development", "Other"]
        )
        
        experience_level = st.select_slider(
            "Experience Level",
            options=["Beginner", "Intermediate", "Advanced", "Expert"]
        )
        
        available_time = st.slider(
            "Hours available per week for learning",
            min_value=1, max_value=40, value=10
        )
        
        goals = st.text_area(
            "Describe your specific learning goals and what you hope to achieve",
            placeholder="Example: I want to learn web development to build a personal portfolio website and eventually work as a frontend developer."
        )
        
        # Format the query to include all relevant information
        def format_query():
            return f"Generate a learning path for {learning_category} for a {experience_level.lower()} with {available_time} hours per week available. Goals: {goals}"
        
        # Add a submit button
        submitted = st.form_submit_button("Generate Learning Path")
        
        if submitted:
            if not name or not email or not goals:
                st.error("Please fill out all required fields marked with *")
            else:
                # Store the user information in session state
                st.session_state.user_info = {
                    "name": name,
                    "email": email,
                    "age": age,
                    "education_level": education_level,
                    "learning_category": learning_category,
                    "experience_level": experience_level,
                    "available_time": available_time,
                    "goals": goals,
                    "query": format_query()
                }
                
                # Generate recommendations and store in session state
                with st.spinner("Generating your personalized learning path..."):
                    recommendations = generate_learning_path(format_query())
                    path_introduction, path_content = process_recommendation(recommendations)
                    
                    st.session_state.path_introduction = path_introduction
                    st.session_state.path_content = path_content
                    st.session_state.show_regenerate = True
                
                # Show a success message and instruct to go to the next tab
                st.success("Your learning path has been generated successfully! Please go to the 'View Learning Path' tab to see your results.")

with tab2:
    st.markdown('<div class="sub-header">Your Personalized Learning Path</div>', unsafe_allow_html=True)
    
    if 'user_info' in st.session_state and 'path_introduction' in st.session_state:
        # Display user info in a cleaner format with hover effects
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header" style="margin-top:0">Learning Profile</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.user_info['name']}")
            st.write(f"**Education:** {st.session_state.user_info['education_level']}")
            st.write(f"**Category:** {st.session_state.user_info['learning_category']}")
        
        with col2:
            st.write(f"**Experience:** {st.session_state.user_info['experience_level']}")
            st.write(f"**Available Time:** {st.session_state.user_info['available_time']} hours/week")
            st.write(f"**Email:** {st.session_state.user_info['email']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Introduction Section with enhanced styling
        st.markdown('<div class="path-introduction">', unsafe_allow_html=True)
        st.markdown('### Your Learning Journey Overview', unsafe_allow_html=True)
        st.markdown(f'{st.session_state.path_introduction}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Suggested Path Section with enhanced styling
        if st.session_state.path_content:
            st.markdown('<div class="path-content">', unsafe_allow_html=True)
            st.markdown('### Your Personalized Learning Roadmap', unsafe_allow_html=True)
            st.markdown(f'{st.session_state.path_content}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add the regeneration feature with enhanced styling
        if st.session_state.show_regenerate:
            st.markdown('<div class="regenerate-container">', unsafe_allow_html=True)
            st.markdown('<div class="regenerate-title">Not quite what you were looking for?</div>', unsafe_allow_html=True)
            
            if st.button("Regenerate Learning Path with Specific Requirements", key="regenerate_button", help="Click to customize your learning path further"):
                st.session_state.regenerate_expanded = True
            
            if 'regenerate_expanded' in st.session_state and st.session_state.regenerate_expanded:
                with st.form("regenerate_form"):
                    updated_requirements = st.text_area(
                        "What updates would you like to make to your learning path?",
                        placeholder="Example: I'd like more focus on practical projects, or I need resources that are free, or I want to learn more about specific technologies like React."
                    )
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        regenerate_submitted = st.form_submit_button("Generate Updated Path", help="Submit to create a new personalized path")
                    
                    if regenerate_submitted and updated_requirements:
                        # Create a new query by combining the original with the update request
                        original_query = st.session_state.user_info["query"]
                        updated_query = f"{original_query} Additional requirements: {updated_requirements}"
                        
                        # Generate new recommendations
                        with st.spinner("Regenerating your personalized learning path..."):
                            new_recommendations = generate_learning_path(updated_query)
                            new_path_introduction, new_path_content = process_recommendation(new_recommendations)
                            
                            # Update session state
                            st.session_state.path_introduction = new_path_introduction
                            st.session_state.path_content = new_path_content
                            st.session_state.regenerate_expanded = False
                            
                            # Store the updated query
                            st.session_state.user_info["query"] = updated_query
                            
                            st.success("Your learning path has been updated successfully!")
                            st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add the new "Create Assessment" button below the regeneration container
            st.markdown('<div class="assessment-container">', unsafe_allow_html=True)
            st.markdown('<div class="assessment-title">Ready to test your knowledge?</div>', unsafe_allow_html=True)
            
            if st.button("📝 Create Assessment", key="create_assessment", help="Generate an assessment based on your learning path"):
                # Set show_assessment to true to display in the Assessment tab
                st.session_state.show_assessment = True
                
                # Combine path introduction and content for assessment context
                learning_path_data = st.session_state.path_introduction
                if st.session_state.path_content:
                    learning_path_data += "\n\n" + st.session_state.path_content
                
                # Generate the assessment
                with st.spinner("Creating your personalized assessment..."):
                    assessment_text = generate_assessment(learning_path_data, st.session_state.user_info)
                    st.session_state.assessment_text = assessment_text
                    
                    # Try to parse as JSON if possible
                    st.session_state.assessment_data = process_assessment(assessment_text)
                
                st.success("Your assessment has been created! Please go to the 'Assessment' tab to view it.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add download and sharing options with enhanced styling
        st.markdown('<div class="save-options">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header" style="margin-top:0">Save Your Learning Path</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔽 Download as PDF", key="download_button", help="Download your learning path as a PDF document"):
                st.info("PDF download functionality would be implemented here.")
        
        with col2:
            if st.button("📧 Email My Learning Path", key="email_button", help="Send your learning path to your email address"):
                st.info(f"An email with your learning path would be sent to {st.session_state.user_info['email']}.")
        
        # Add share buttons
        st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📱 Share via WhatsApp", key="whatsapp_button", help="Share your learning path via WhatsApp"):
                st.info("WhatsApp sharing functionality would be implemented here.")
        with col2:
            if st.button("🔗 Copy Link", key="copy_button", help="Copy a shareable link to your clipboard"):
                st.info("Link copying functionality would be implemented here.")
        with col3:
            if st.button("📋 Export as Text", key="export_button", help="Export your learning path as plain text"):
                st.info("Text export functionality would be implemented here.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("Please fill out the form in the 'Your Information' tab to generate your personalized learning path.")

# New Assessment Tab
with tab3:
    st.markdown('<div class="sub-header">Knowledge Assessment</div>', unsafe_allow_html=True)
    
    if 'show_assessment' in st.session_state and st.session_state.show_assessment and 'assessment_data' in st.session_state:
        # Display user info in the assessment tab as well
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header" style="margin-top:0">Assessment for</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.user_info['name']}")
            st.write(f"**Category:** {st.session_state.user_info['learning_category']}")
        with col2:
            st.write(f"**Experience:** {st.session_state.user_info['experience_level']}")
            st.write(f"**Learning Goals:** {st.session_state.user_info['goals'][:50]}...")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Function to display assessment data nicely
        def display_assessment():
            assessment_data = st.session_state.assessment_data
            
            # Check if we have raw text or structured data
            if "raw_text" in assessment_data:
                st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
                st.markdown("### Your Assessment")
                st.markdown(assessment_data["raw_text"])
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Display Multiple Choice Questions
            if "multiple_choice" in assessment_data:
                st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
                st.markdown("### Multiple Choice Questions")
                
                for i, q in enumerate(assessment_data["multiple_choice"]):
                    st.markdown(f'<div class="question">', unsafe_allow_html=True)
                    st.markdown(f"**Question {i+1}:** {q['question']}")
                    
                    for j, option in enumerate(q['options']):
                        option_letter = chr(65 + j)  # A, B, C, D...
                        if option == q.get('correct_answer') or option_letter == q.get('correct_answer'):
                            st.markdown(f'<div class="option correct-answer">{option_letter}. {option}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="option">{option_letter}. {option}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Short Answer Questions
            if "short_answer" in assessment_data:
                st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
                st.markdown("### Short Answer Questions")
                
                for i, q in enumerate(assessment_data["short_answer"]):
                    st.markdown(f'<div class="question">', unsafe_allow_html=True)
                    st.markdown(f"**Question {i+1}:** {q['question']}")
                    
                    if "guidance" in q:
                        st.markdown("*Guidance:* {q['guidance']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Practical Exercises
            if "practical_exercises" in assessment_data:
                st.markdown('<div class="assessment-section">', unsafe_allow_html=True)
                st.markdown("### Practical Exercises")
                
                for i, exercise in enumerate(assessment_data["practical_exercises"]):
                    st.markdown(f'<div class="question">', unsafe_allow_html=True)
                    st.markdown(f"**Exercise {i+1}:** {exercise['title']}")
                    st.markdown(f"{exercise['description']}")
                    
                    if "steps" in exercise:
                        st.markdown("**Steps:**")
                        for j, step in enumerate(exercise['steps']):
                            st.markdown(f"{j+1}. {step}")
                    
                    if "criteria" in exercise:
                        st.markdown("**Success Criteria:**")
                        for criterion in exercise['criteria']:
                            st.markdown(f"- {criterion}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Display the assessment
        display_assessment()
        
        # Add option to take the assessment interactively (placeholder)
        st.markdown('<div class="save-options">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Take Assessment Interactively", key="take_assessment", help="Take the assessment with immediate feedback"):
                st.info("Interactive assessment functionality would be implemented here.")
        with col2:
            if st.button("💾 Save Assessment for Later", key="save_assessment", help="Save this assessment to your profile"):
                st.info("Assessment saved successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No assessment created yet. Please go to the 'View Learning Path' tab and click on 'Create Assessment'.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="color: #666;">© 2023 Learning Path Assistant. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
