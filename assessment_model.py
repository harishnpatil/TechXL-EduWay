import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('new.env')

class AssessmentGenerator:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize the model
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                google_api_key=self.gemini_api_key
            )
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            print("Trying fallback model...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
                google_api_key=self.gemini_api_key
            )
    
    def generate_assessment(self, learning_path_data, user_info):
        """
        Generate an assessment based on the learning path and user information.
        
        Args:
            learning_path_data (str): The content of the learning path
            user_info (dict): User information including experience level, goals, etc.
            
        Returns:
            dict: Assessment with sections for different question types
        """
        # Extract skills and topics from the learning path
        topics = self._extract_topics(learning_path_data)
        
        # Create prompt for assessment generation
        prompt_template = PromptTemplate(
            template="""
            You are an expert education assessment creator. Create an assessment for a student with the following profile:
            
            Name: {name}
            Experience Level: {experience_level}
            Learning Category: {category}
            Goals: {goals}
            
            They have been studying the following topics/learning paths:
            {topics}
            
            Create a comprehensive assessment with the following sections:
            
            1. Multiple Choice Questions (5 questions):
               - Create 5 multiple-choice questions with 4 options each
               - Include questions of varying difficulty based on the user's experience level
               - Each question should test understanding of key concepts from the learning path
               - Indicate the correct answer
            
            2. Short Answer Questions (3 questions):
               - Create 3 questions that require brief explanations
               - These should test deeper understanding and application of concepts
               - Include a brief guideline on what constitutes a good answer
            
            3. Practical Exercise (1-2 exercises):
               - Design 1-2 hands-on exercises related to the learning path
               - The exercises should be appropriate for the user's experience level ({experience_level})
               - Include clear instructions, requirements, and evaluation criteria
               - For coding topics, include starter code or templates if appropriate
            
            4. Self-Assessment Reflection (3 questions):
               - Create 3 reflection questions to help the user assess their own understanding
               - These should encourage critical thinking about what they've learned
            
            Format your response as a structured assessment with clear sections, instructions, and question numbering.
            Make sure the assessment is challenging but appropriate for a {experience_level} level learner.
            
            Return the results in a JSON format with four keys: 'multiple_choice', 'short_answer', 'practical_exercise', and 'self_assessment'.
            """,
            input_variables=["name", "experience_level", "category", "goals", "topics"]
        )
        
        # Prepare the input for the prompt
        input_data = {
            "name": user_info.get("name", "Student"),
            "experience_level": user_info.get("experience_level", "Beginner"),
            "category": user_info.get("learning_category", "General"),
            "goals": user_info.get("goals", "Learning new skills"),
            "topics": topics
        }
        
        # Generate the assessment
        prompt = prompt_template.format(**input_data)
        
        try:
            response = self.llm.invoke(prompt)
            assessment_text = response.content
            
            # Process the response to extract JSON content
            # Note: We'll handle non-JSON responses properly in the UI
            return assessment_text
        except Exception as e:
            return f"Error generating assessment: {str(e)}"
    
    def _extract_topics(self, learning_path_data):
        """
        Extract relevant topics from the learning path data.
        
        Args:
            learning_path_data (str): The content of the learning path
            
        Returns:
            str: Extracted topics as a string
        """
        # For simplicity, we'll just return the learning path data
        # In a more sophisticated implementation, you might use regex or other methods
        # to extract just the topic names or course titles
        return learning_path_data

    def evaluate_user_answers(self, assessment, user_answers):
        """
        Evaluate user answers against the assessment.
        
        Args:
            assessment (dict): The assessment with questions and correct answers
            user_answers (dict): The user's submitted answers
            
        Returns:
            dict: Evaluation results with feedback and score
        """
        try:
            # Create prompt for answer evaluation
            prompt_template = PromptTemplate(
                template="""
                You are an expert education assessment evaluator. Evaluate the user's answers for the following assessment:
                
                Assessment: 
                {assessment}
                
                User's Answers:
                {user_answers}
                
                Provide detailed feedback for each answer, indicating what was correct and what could be improved.
                For multiple choice questions, mark each as correct or incorrect.
                For short answer questions, provide constructive feedback.
                For practical exercises, evaluate based on the specified criteria.
                
                Calculate an overall score as a percentage.
                
                Return the results in a JSON format with the following structure:
                {{
                    "score": 85,
                    "feedback": {{
                        "multiple_choice": [...],
                        "short_answer": [...],
                        "practical_exercise": [...],
                        "self_assessment": [...]
                    }},
                    "strengths": ["..."],
                    "areas_for_improvement": ["..."],
                    "recommendations": ["..."]
                }}
                """,
                input_variables=["assessment", "user_answers"]
            )
            
            # Prepare the input for the prompt
            input_data = {
                "assessment": str(assessment),
                "user_answers": str(user_answers)
            }
            
            # Generate the evaluation
            prompt = prompt_template.format(**input_data)
            
            response = self.llm.invoke(prompt)
            evaluation_text = response.content
            
            # Process the response to extract JSON content
            # Here we would implement JSON extraction logic similar to what's in the Streamlit app
            return evaluation_text
        except Exception as e:
            return f"Error evaluating answers: {str(e)}"

def generate_assessment(learning_path_data, user_info):
    """
    Generate an assessment based on a learning path.
    
    Args:
        learning_path_data (str): The learning path content
        user_info (dict): User information
        
    Returns:
        str: The generated assessment
    """
    try:
        generator = AssessmentGenerator()
        return generator.generate_assessment(learning_path_data, user_info)
    except Exception as e:
        return f"Error generating assessment: {str(e)}"

def evaluate_user_answers(assessment, user_answers):
    """
    Evaluate user answers against the assessment.
    
    Args:
        assessment (dict): The assessment with questions and correct answers
        user_answers (dict): The user's submitted answers
        
    Returns:
        dict: Evaluation results with feedback and score
    """
    try:
        generator = AssessmentGenerator()
        return generator.evaluate_user_answers(assessment, user_answers)
    except Exception as e:
        return f"Error evaluating answers: {str(e)}"