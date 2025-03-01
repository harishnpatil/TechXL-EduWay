import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from recommendation_model import generate_learning_path, GenerateLearningPathIndexEmbeddings

# Function to check and update the FAISS index
def update_faiss_index(csv_filename):
    try:
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
    except Exception as e:
        st.error(f"Error updating FAISS index: {str(e)}")

# Function to split response into introduction and table
def process_recommendation(recommendation_text):
    try:
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
    except Exception as e:
        st.error(f"Error processing recommendation: {str(e)}")
        return recommendation_text, ""

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

# Initialize session state variables if they don't exist
if 'show_regenerate' not in st.session_state:
    st.session_state.show_regenerate = False

# Create a cleaner form with tabs
tab1, tab2 = st.tabs(["Your Information", "View Learning Path"])

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
                
                # Update FAISS index before generating recommendations
                try:
                    update_faiss_index(csv_filename)
                    
                    # Generate recommendations and store in session state
                    with st.spinner("Generating your personalized learning path..."):
                        recommendations = generate_learning_path(format_query())
                        path_introduction, path_content = process_recommendation(recommendations)
                        
                        st.session_state.path_introduction = path_introduction
                        st.session_state.path_content = path_content
                        st.session_state.show_regenerate = True
                    
                    # Show a success message and instruct to go to the next tab
                    st.success("Your learning path has been generated successfully! Please go to the 'View Learning Path' tab to see your results.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

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
                            try:
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
                            except Exception as e:
                                st.error(f"Error regenerating learning path: {str(e)}")
            
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
