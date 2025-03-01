import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from recommendation_model import generate_learning_path, GenerateLearningPathIndexEmbeddings  # Import the class

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
        
        introduction = parts[0].strip()
        table = '| Learning Pathway | duration | link | Module |\n' + parts[1].strip()
        
        return introduction, table
    else:
        # If table format isn't found, return the whole text as introduction
        return recommendation_text, ""

# Set the title of the app with improved styling
st.set_page_config(page_title="Learning Path Assistant", layout="wide")

# Custom CSS for better styling
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
    .table-container {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        overflow-x: auto;
    }
    .regenerate-container {
        margin-top: 1.5rem;
        background-color: #EDE7F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #7E57C2;
    }
    .regenerate-title {
        font-size: 1.2rem;
        color: #5E35B1;
        margin-bottom: 0.8rem;
    }
    
    /* New styles for the View Learning Path tab */
    .profile-card {
        background-color: #F5F9FF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
        border-left: 4px solid #1E88E5;
    }
    .profile-card:hover {
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }
    .profile-title {
        font-size: 1.3rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .profile-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #E0E0E0;
        transition: background-color 0.2s ease;
    }
    .profile-item:hover {
        background-color: #EEF5FF;
    }
    .profile-label {
        font-weight: 600;
        color: #424242;
    }
    .profile-value {
        color: #1976D2;
    }
    .intro-section {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #1E88E5;
    }
    .path-section {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #43A047;
    }
    .section-title {
        color: #0D47A1;
        font-size: 1.4rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .save-btn {
        background-color: #1976D2;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .save-btn:hover {
        background-color: #0D47A1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    .email-btn {
        background-color: #43A047;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .email-btn:hover {
        background-color: #2E7D32;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Table styling for learning path */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .styled-table th {
        background-color: #1E88E5;
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    .styled-table th:first-child {
        border-top-left-radius: 8px;
    }
    .styled-table th:last-child {
        border-top-right-radius: 8px;
    }
    .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #E0E0E0;
    }
    .styled-table tr {
        transition: all 0.3s ease;
    }
    .styled-table tr:hover {
        background-color: #F5F9FF;
        transform: scale(1.01);
    }
    .styled-table tr:last-child td:first-child {
        border-bottom-left-radius: 8px;
    }
    .styled-table tr:last-child td:last-child {
        border-bottom-right-radius: 8px;
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
                
                # Generate recommendations and store in session state
                with st.spinner("Generating your personalized learning path..."):
                    recommendations = generate_learning_path(format_query())
                    introduction, table = process_recommendation(recommendations)
                    
                    st.session_state.introduction = introduction
                    st.session_state.table = table
                    st.session_state.show_regenerate = True
                
                # Show a success message and instruct to go to the next tab
                st.success("Your learning path has been generated successfully! Please go to the 'View Learning Path' tab to see your results.")

with tab2:
    st.markdown('<div class="sub-header">Your Personalized Learning Path</div>', unsafe_allow_html=True)
    
    if 'user_info' in st.session_state and 'introduction' in st.session_state:
        # Display user info in a nicer profile card
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown('<div class="profile-title">Learning Profile</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="profile-item"><span class="profile-label">Name:</span> <span class="profile-value">{st.session_state.user_info["name"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-item"><span class="profile-label">Education:</span> <span class="profile-value">{st.session_state.user_info["education_level"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-item"><span class="profile-label">Category:</span> <span class="profile-value">{st.session_state.user_info["learning_category"]}</span></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="profile-item"><span class="profile-label">Experience:</span> <span class="profile-value">{st.session_state.user_info["experience_level"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="profile-item"><span class="profile-label">Available Time:</span> <span class="profile-value">{st.session_state.user_info["available_time"]} hours/week</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Introduction Section
        st.markdown('<div class="intro-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Your Learning Journey</div>', unsafe_allow_html=True)
        st.markdown(f'{st.session_state.introduction}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Suggested Path Section with styled table
        if st.session_state.table:
            st.markdown('<div class="path-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Recommended Learning Path</div>', unsafe_allow_html=True)
            
            # Convert the Markdown table to an HTML table with our custom styling
            table_lines = st.session_state.table.strip().split('\n')
            if len(table_lines) >= 2:
                headers = [header.strip() for header in table_lines[0].split('|') if header.strip()]
                
                # Start building HTML table
                html_table = '<table class="styled-table"><thead><tr>'
                
                # Add table headers
                for header in headers:
                    html_table += f'<th>{header}</th>'
                
                html_table += '</tr></thead><tbody>'
                
                # Add table rows
                for i in range(2, len(table_lines)):
                    row = table_lines[i]
                    if '|' in row:  # Make sure it's a valid table row
                        cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                        html_table += '<tr>'
                        
                        for cell in cells:
                            html_table += f'<td>{cell}</td>'
                        
                        html_table += '</tr>'
                
                html_table += '</tbody></table>'
                
                # Display the styled HTML table
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                # Fallback to original markdown table if parsing fails
                st.markdown(f'{st.session_state.table}', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add the regeneration feature
        if st.session_state.show_regenerate:
            st.markdown('<div class="regenerate-container">', unsafe_allow_html=True)
            st.markdown('<div class="regenerate-title">Not quite what you were looking for?</div>', unsafe_allow_html=True)
            
            if st.button("Regenerate Learning Path with Specific Requirements"):
                st.session_state.regenerate_expanded = True
            
            if 'regenerate_expanded' in st.session_state and st.session_state.regenerate_expanded:
                with st.form("regenerate_form"):
                    updated_requirements = st.text_area(
                        "What updates would you like to make to your learning path?",
                        placeholder="Example: I'd like more focus on practical projects, or I need resources that are free, or I want to learn more about specific technologies like React."
                    )
                    
                    regenerate_submitted = st.form_submit_button("Generate Updated Path")
                    
                    if regenerate_submitted and updated_requirements:
                        # Create a new query by combining the original with the update request
                        original_query = st.session_state.user_info["query"]
                        updated_query = f"{original_query} Additional requirements: {updated_requirements}"
                        
                        # Generate new recommendations
                        with st.spinner("Regenerating your personalized learning path..."):
                            new_recommendations = generate_learning_path(updated_query)
                            new_introduction, new_table = process_recommendation(new_recommendations)
                            
                            # Update session state
                            st.session_state.introduction = new_introduction
                            st.session_state.table = new_table
                            st.session_state.regenerate_expanded = False
                            
                            # Store the updated query
                            st.session_state.user_info["query"] = updated_query
                            
                            st.success("Your learning path has been updated successfully!")
                            st.experimental_rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced download and sharing options with custom styled buttons
        st.markdown('<div class="sub-header">Save Your Learning Path</div>', unsafe_allow_html=True)
        
        # Create a container for our buttons
        st.markdown('<div style="display: flex; justify-content: center; margin-top: 1.5rem;">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="save-btn" id="pdf-btn" onclick="alert(\'Download PDF feature would be implemented here\');">'
                f'<i style="margin-right: 8px;">📥</i> Download as PDF'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button("Download PDF", key="pdf_btn_backend"):
                st.success("Your learning path PDF is being generated!")
                st.info("PDF download functionality would be implemented here.")
        
        with col2:
            st.markdown(
                f'<div class="email-btn" id="email-btn" onclick="alert(\'Email sent to {st.session_state.user_info[\"email\"]}\');">'
                f'<i style="margin-right: 8px;">📧</i> Email My Learning Path'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button("Email Path", key="email_btn_backend"):
                st.success(f"✅ Learning path successfully sent to {st.session_state.user_info['email']}!")
                st.info("Email functionality would be implemented here.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a progress tracker section
        st.markdown('<div style="margin-top: 2rem; background-color: #F0F7FF; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #FFA000;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.3rem; color: #FFA000; margin-bottom: 1rem; font-weight: 600;">Track Your Progress</div>', unsafe_allow_html=True)
        st.markdown('<div style="color: #424242; margin-bottom: 1rem;">Keep track of your learning journey to stay motivated and see how far you\'ve come!</div>', unsafe_allow_html=True)
        
        # Sample progress bar
        progress_value = 0
        if 'progress' in st.session_state:
            progress_value = st.session_state.progress
        
        progress = st.slider("Your Progress", 0, 100, progress_value, key="progress_slider")
        st.session_state.progress = progress
        
        # Motivational message based on progress
        if progress < 25:
            st.markdown('<div style="color: #2196F3; font-weight: 500; margin-top: 0.5rem;">You\'re just getting started! Keep going!</div>', unsafe_allow_html=True)
        elif progress < 50:
            st.markdown('<div style="color: #2196F3; font-weight: 500; margin-top: 0.5rem;">Great progress! You\'re building momentum!</div>', unsafe_allow_html=True)
        elif progress < 75:
            st.markdown('<div style="color: #2196F3; font-weight: 500; margin-top: 0.5rem;">You\'re doing amazing! Keep up the good work!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: #2196F3; font-weight: 500; margin-top: 0.5rem;">Incredible work! You\'re almost there!</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("Please fill out the form in the 'Your Information' tab to generate your personalized learning path.")