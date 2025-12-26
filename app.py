import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from PyPDF2 import PdfReader as PyPDF2Reader
from st_aggrid import AgGrid, GridOptionsBuilder
import mysql.connector
import json
from datetime import datetime

# --- Database Configuration ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "root"  # Leave it empty if your database has no password
DB_NAME = "rankitright_db"


def create_connection():
    try:
        mydb = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return mydb
    except mysql.connector.Error as err:
        st.error(f"Error connecting to database: {err}")
        return None


# --- User Authentication ---
def create_user(username, password, role):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO Users (Username, Password, Role) VALUES (%s, %s, %s)"
            cursor.execute(sql, (username, password, role))  # In a real app, hash the password!
            mydb.commit()
            st.success("User created successfully!")
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error creating user: {err}")
            mydb.close()
            return False


def verify_user(username, password):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT UserID, Role FROM Users WHERE Username = %s AND Password = %s"  # In a real app, compare hashed passwords!
            cursor.execute(sql, (username, password))
            result = cursor.fetchone()
            mydb.close()
            if result:
                return result[0], result[1]
            else:
                return None, None
        except mysql.connector.Error as err:
            st.error(f"Error verifying user: {err}")
            mydb.close()
            return None, None


def save_hr_ranking_history(user_id, job_description, resumes, scores):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO HRResumeRankingHistory (UserID, JobDescription, Resumes, Scores) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (user_id, job_description, ",".join(resumes), ",".join(map(str, scores))))
            mydb.commit()
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error saving HR ranking history: {err}")
            mydb.close()
            return False


def save_hr_soft_skill_history(user_id, videos, scores):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO HRSoftSkillRankingHistory (UserID, Videos, Scores) VALUES (%s, %s, %s)"
            cursor.execute(sql, (user_id, ",".join(videos), json.dumps(scores)))
            mydb.commit()
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error saving HR soft skill history: {err}")
            mydb.close()
            return False


def save_hr_feedback(user_id, feedback):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO HRFeedbackHistory (UserID, Feedback) VALUES (%s, %s)"
            cursor.execute(sql, (user_id, feedback))
            mydb.commit()
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error saving HR feedback: {err}")
            mydb.close()
            return False


def get_hr_ranking_history(user_id):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT JobDescription, Resumes, Scores, Timestamp FROM HRResumeRankingHistory WHERE UserID = %s ORDER BY Timestamp DESC"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            mydb.close()
            return results
        except mysql.connector.Error as err:
            st.error(f"Error fetching HR ranking history: {err}")
            mydb.close()
            return []


def get_hr_soft_skill_history(user_id):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT Videos, Scores, Timestamp FROM HRSoftSkillRankingHistory WHERE UserID = %s ORDER BY Timestamp DESC"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            mydb.close()
            return results
        except mysql.connector.Error as err:
            st.error(f"Error fetching HR soft skill history: {err}")
            mydb.close()
            return []


def get_hr_feedback_history(user_id):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT Feedback, Timestamp FROM HRFeedbackHistory WHERE UserID = %s ORDER BY Timestamp DESC"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            mydb.close()
            return results
        except mysql.connector.Error as err:
            st.error(f"Error fetching HR feedback history: {err}")
            mydb.close()
            return []


def save_student_resume_check_history(user_id, filename, suggestions):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO StudentResumeCheckHistory (UserID, Filename, Suggestions) VALUES (%s, %s, %s)"
            cursor.execute(sql, (user_id, filename, ",".join(suggestions)))
            mydb.commit()
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error saving student resume check history: {err}")
            mydb.close()
            return False


def get_student_resume_check_history(user_id):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT Filename, Suggestions, Timestamp FROM StudentResumeCheckHistory WHERE UserID = %s ORDER BY Timestamp DESC"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            mydb.close()
            return results
        except mysql.connector.Error as err:
            st.error(f"Error fetching student resume check history: {err}")
            mydb.close()
            return []


def save_student_feedback(user_id, feedback):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "INSERT INTO StudentFeedbackHistory (UserID, Feedback) VALUES (%s, %s)"
            cursor.execute(sql, (user_id, feedback))
            mydb.commit()
            mydb.close()
            return True
        except mysql.connector.Error as err:
            st.error(f"Error saving student feedback: {err}")
            mydb.close()
            return False


def get_student_feedback_history(user_id):
    mydb = create_connection()
    if mydb:
        cursor = mydb.cursor()
        try:
            sql = "SELECT Feedback, Timestamp FROM StudentFeedbackHistory WHERE UserID = %s ORDER BY Timestamp DESC"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            mydb.close()
            return results
        except mysql.connector.Error as err:
            st.error(f"Error fetching student feedback history: {err}")
            mydb.close()
            return []


# --- HR Code ---
def extract_text_from_pdf_hr(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"No text found on page {page.page_number} of {file.name} (HR).")
    except Exception as e:
        st.error(f"Error reading {file.name} (HR): {e}")
    return text


def rank_resumes_hr(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities


def hr_resume_ranking_app(user_id):
    st.subheader("Resume Ranking")
    job_description = st.text_area("Enter the job description for HR", height=200)
    uploaded_files = st.file_uploader("Upload PDF resumes for ranking", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        resumes_text = []
        resume_names = [file.name for file in uploaded_files]
        with st.spinner("Processing resumes..."):
            for file in uploaded_files:
                try:
                    text = extract_text_from_pdf_hr(file)
                    resumes_text.append(text)
                except Exception as e:
                    st.error(f"Error extracting text from {file.name}: {e}")
                    return

            if resumes_text:
                scores = rank_resumes_hr(job_description, resumes_text)
                ranked_indices = scores.argsort()[::-1]
                ranked_scores = scores[ranked_indices]
                ranked_names = [resume_names[i] for i in ranked_indices]

                results_df = pd.DataFrame({
                    "Resume": ranked_names,
                    "Score": ranked_scores.round(2)
                })

                st.success("Resumes ranked successfully!")
                st.subheader("Ranking Results")
                gb = GridOptionsBuilder.from_dataframe(results_df)
                gb.configure_columns(['Score'], type=['numericColumnFilter', 'customNumericFormat'], precision=2)
                gridOptions = gb.build()
                AgGrid(results_df, gridOptions=gridOptions, height=300, fit_columns_on_grid_load=True)

                fig, ax = plt.subplots()
                ax.pie(ranked_scores, labels=ranked_names, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                save_hr_ranking_history(user_id, job_description, resume_names, ranked_scores.tolist())

    elif uploaded_files:
        st.warning("Please enter a job description to rank the resumes.")
    elif job_description:
        st.warning("Please upload resumes to perform ranking.")
    else:
        st.info("Upload resumes and enter a job description to see the ranking.")


def hr_soft_skill_ranking_app(user_id):
    st.subheader("Soft Skill Ranking")
    st.info("Upload interview videos for analysis based on communication, tone, and confidence.")
    uploaded_videos = st.file_uploader("Upload Video files for soft skill analysis", type=["mp4", "avi", "mov"],
                                         accept_multiple_files=True)

    if uploaded_videos:
        st.info("Note: Soft skill analysis is a complex task and this is a simplified simulation.")
        with st.spinner("Analyzing videos..."):
            video_names = [video.name for video in uploaded_videos]
            communication_scores = np.random.uniform(0.6, 0.95, len(uploaded_videos)).round(2)
            tone_scores = np.random.uniform(0.55, 0.9, len(uploaded_videos)).round(2)
            confidence_scores = np.random.uniform(0.7, 1.0, len(uploaded_videos)).round(2)

            results_df = pd.DataFrame({
                "Video Name": video_names,
                "Communication": communication_scores,
                "Tone": tone_scores,
                "Confidence": confidence_scores
            })
            results_df['Combined Score'] = ((communication_scores + tone_scores + confidence_scores) / 3).round(2)
            ranked_results = results_df.sort_values(by='Combined Score', ascending=False).reset_index(drop=True)
            ranked_results.index += 1

            st.success("Soft skill analysis complete!")
            st.subheader("Soft Skill Ranking Results")
            gb = GridOptionsBuilder.from_dataframe(ranked_results)
            gb.configure_columns(['Communication', 'Tone', 'Confidence', 'Combined Score'],
                                  type=['numericColumnFilter', 'customNumericFormat'], precision=2)
            gridOptions = gb.build()
            AgGrid(ranked_results, gridOptions=gridOptions, height=350, fit_columns_on_grid_load=True)

            st.subheader("Detailed Scores")
            st.bar_chart(ranked_results.set_index("Video Name")[["Communication", "Tone", "Confidence"]])

            scores_data = ranked_results[['Video Name', 'Communication', 'Tone', 'Confidence', 'Combined Score']].to_dict(
                'records')
            save_hr_soft_skill_history(user_id, video_names, scores_data)
    else:
        st.info("Upload interview videos to analyze soft skills.")


def hr_feedback_app(user_id):
    st.subheader("Feedback")
    feedback = st.text_area("Please provide your feedback here:", height=150)
    if st.button("Submit Feedback", key="hr_submit_feedback", use_container_width=True):
        if feedback:
            if save_hr_feedback(user_id, feedback):
                st.success("Thank you for your feedback!")
                st.session_state["hr_feedback_submitted"] = True
            else:
                st.error("Failed to submit feedback.")
        else:
            st.error("Please enter your feedback before submitting.")

    if "hr_feedback_submitted" in st.session_state and st.session_state["hr_feedback_submitted"]:
        st.balloons()
        del st.session_state["hr_feedback_submitted"]



def hr_manage_history_app(user_id):
    st.subheader("Manage History")

    with st.expander("Resume Ranking History", expanded=True):
        ranking_history = get_hr_ranking_history(user_id)
        if ranking_history:
            simplified_history = []
            for jd, resumes_str, scores_str, timestamp in ranking_history:
                resumes = resumes_str.split(',')
                scores = [float(s) for s in scores_str.split(',')] if scores_str else []
                simplified_history.append({
                    "Action": "Resume Ranking",
                    "Job Description": jd[:50] + "...",
                    "Resumes": ", ".join(resumes),
                    "Avg. Score": f"{np.mean(scores).round(2):.2f}" if scores else "N/A",
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                })
            if simplified_history:
                history_df = pd.DataFrame(simplified_history)
                history_df.index += 1
                AgGrid(history_df, height=300, fit_columns_on_grid_load=True)
            else:
                st.info("No resume ranking history available.")
        else:
            st.info("No resume ranking history available.")

    with st.expander("Soft Skill Ranking History", expanded=True):
        soft_skill_history = get_hr_soft_skill_history(user_id)
        if soft_skill_history:
            simplified_history = []
            for videos_str, scores_json, timestamp in soft_skill_history:
                videos = videos_str.split(',')
                scores = json.loads(scores_json) if scores_json else []
                avg_combined_score = np.mean([rec['Combined Score'] for rec in scores] or [0]).round(2) if scores else "N/A"
                simplified_history.append({
                    "Action": "Soft Skill Ranking",
                    "Videos": ", ".join(videos),
                    "Avg. Combined Score": avg_combined_score,
                    "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                })
            if simplified_history:
                history_df = pd.DataFrame(simplified_history)
                history_df.index += 1
                AgGrid(history_df, height=300, fit_columns_on_grid_load=True)
            else:
                st.info("No soft skill ranking history available.")
        else:
            st.info("No soft skill ranking history available.")

    with st.expander("Feedback History", expanded=True):
        feedback_history = get_hr_feedback_history(user_id)
        if feedback_history:
            feedback_df = pd.DataFrame(feedback_history, columns=["Feedback", "Timestamp"])
            feedback_df['Timestamp'] = feedback_df['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
            feedback_df.index += 1
            AgGrid(feedback_df, height=200, fit_columns_on_grid_load=True)
        else:
            st.info("No feedback history available.")


def hr_chatbot_app():
    st.subheader("HR Chatbot")
    st.info("Ask me questions related to recruitment, resume screening, or soft skills assessment.")

    if "hr_chatbot_messages" not in st.session_state:
        st.session_state["hr_chatbot_messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    for msg in st.session_state["hr_chatbot_messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Your question"):
        st.session_state["hr_chatbot_messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Simulate chatbot response (replace with actual logic)
        import time
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)
            response = f"This is a simulated response to your question: '{prompt}'. For real-time and accurate information, please refer to the relevant documentation or consult with a senior HR professional."
            st.session_state["hr_chatbot_messages"].append({"role": "assistant", "content": response})
            st.write(response)



# --- Student Code ---
def extract_text_from_pdf_student(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"No text found on page {page.page_number} of {file.name} (Student).")
    except Exception as e:
        st.error(f"Error reading {file.name} (Student): {e}")
    return text



def evaluate_resume_student(text):
    suggestions = []
    text_lower = text.lower()
    if len(text.split()) < 150:
        suggestions.append("Consider adding more detail to your resume.")
    if "skills" not in text_lower and "technical proficiencies" not in text_lower:
        suggestions.append("Include a 'Skills' or 'Technical Proficiencies' section.")
    if "experience" not in text_lower and "professional history" not in text_lower:
        suggestions.append("Add an 'Experience' or 'Professional History' section detailing your roles and responsibilities.")
    if "education" not in text_lower and "academic background" not in text_lower:
        suggestions.append("Include an 'Education' or 'Academic Background' section with your degrees and institutions.")
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) < 10:
        suggestions.append("Your resume appears to be quite brief; consider elaborating on your experiences and skills.")
    if "objective" not in text_lower and "summary" not in text_lower and "profile" not in text_lower:
        suggestions.append("Consider adding a brief 'Objective', 'Summary', or 'Profile' at the beginning.")
    if not any(char in text for char in "@.") or "phone" not in text_lower and "mobile" not in text_lower:
        suggestions.append("Ensure your contact information (email and phone number) is clearly visible.")
    if any(word in text_lower for word in
            ["internship", "intern", "volunteer", "project", "contribution"]):
        suggestions.append(
            "Highlight your internships, volunteer work, projects, and key contributions with specific details.")
    if "certification" not in text_lower and "certificates" not in text_lower and "licensure" not in text_lower:
        suggestions.append("Consider adding a 'Certifications' or 'Licensure' section if applicable.")
    if any(skill in text_lower for skill in ["microsoft word", "excel", "powerpoint"]):
        suggestions.append("List specific software proficiencies rather than just broad terms.")
    if len([word for word in text.split() if word.isupper() and len(word) > 3]) > 10:
        suggestions.append("Review your use of excessive capitalization.")
    if text.count('.') < 5:
        suggestions.append("Ensure your descriptions use proper sentence structure.")
    return suggestions


def student_resume_checker_app(user_id):
    st.subheader("Resume Checker")
    uploaded_file = st.file_uploader("Upload your PDF resume for checking", type=["pdf"], accept_multiple_files=False)

    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            text = extract_text_from_pdf_student(uploaded_file)
            if text:
                suggestions = evaluate_resume_student(text)

                st.subheader("Analysis Results")
                if suggestions:
                    st.warning("The following suggestions can help improve your resume:")
                    for i, suggestion in enumerate(suggestions):
                        st.markdown(f"- {i + 1}. {suggestion}")
                else:
                    st.success("Your resume looks well-structured based on our current checks!")

                save_student_resume_check_history(user_id, uploaded_file.name, suggestions)
    else:
        st.info("Upload your resume to get improvement suggestions.")



def student_feedback_app(user_id):
    st.subheader("Feedback")
    feedback = st.text_area("Please provide your feedback on the RankItRight platform:", height=150)
    if st.button("Submit Feedback", key="student_submit_feedback", use_container_width=True):
        if feedback:
            if save_student_feedback(user_id, feedback):
                st.success("Thank you for your feedback! We appreciate your input.")
                st.session_state["student_feedback_submitted"] = True
            else:
                st.error("Failed to submit feedback.")
        else:
            st.error("Please enter your feedback before submitting.")

    if "student_feedback_submitted" in st.session_state and st.session_state["student_feedback_submitted"]:
        st.balloons()
        del st.session_state["student_feedback_submitted"]



def student_manage_history_app(user_id):
    st.subheader("Manage History")

    with st.expander("Action History", expanded=True):
        resume_check_history = get_student_resume_check_history(user_id)
        if resume_check_history:
            history_df = pd.DataFrame(resume_check_history, columns=["Filename", "Suggestions", "Timestamp"])
            history_df['Timestamp'] = history_df['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
            history_df.index += 1
            AgGrid(history_df, height=300, fit_columns_on_grid_load=True)
        else:
            st.info("No resume check history available.")

    with st.expander("Feedback History", expanded=True):
        feedback_history = get_student_feedback_history(user_id)
        if feedback_history:
            feedback_df = pd.DataFrame(feedback_history, columns=["Feedback", "Timestamp"])
            feedback_df['Timestamp'] = feedback_df['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
            feedback_df.index += 1
            AgGrid(feedback_df, height=200, fit_columns_on_grid_load=True)
        else:
            st.info("No feedback history available.")



def student_app(user_id, show_page):
    st.header("Student Dashboard")
    st.sidebar.subheader("Navigation")
    if st.sidebar.button("Resume Checker", key="student_resume_checker_btn", use_container_width=True):
        show_page("student_resume_checker")
    if st.sidebar.button("Feedback", key="student_feedback_btn", use_container_width=True):
        show_page("student_feedback")
    if st.sidebar.button("Manage History", key="student_manage_history_btn", use_container_width=True):
        show_page("student_manage_history")

    st.markdown("---")
    st.markdown("### Student Dashboard Actions")

    if st.session_state.student_current_page == "student_resume_checker":
        student_resume_checker_app(user_id)
    elif st.session_state.student_current_page == "student_feedback":
        student_feedback_app(user_id)
    elif st.session_state.student_current_page == "student_manage_history":
        student_manage_history_app(user_id)
    elif st.session_state.student_current_page is None:
        st.info("Welcome to the Student Dashboard! Use the sidebar to navigate.")
        show_page("student_resume_checker")  # Set a default page



def hr_app(user_id, show_page):
    st.header("HR Professional Dashboard")
    st.sidebar.subheader("Navigation")
    if st.sidebar.button("Resume Ranking", key="hr_resume_ranking_btn", use_container_width=True):
        show_page("hr_resume_ranking")
    if st.sidebar.button("Soft Skill Ranking", key="hr_soft_skill_ranking_btn", use_container_width=True):
        show_page("hr_soft_skill_ranking")
    if st.sidebar.button("Feedback", key="hr_feedback_btn", use_container_width=True):
        show_page("hr_feedback")
    if st.sidebar.button("Manage History", key="hr_manage_history_btn", use_container_width=True):
        show_page("hr_manage_history")
    if st.sidebar.button("Chatbot", key="hr_chatbot_btn", use_container_width=True):
        show_page("hr_chatbot")

    st.markdown("---")
    st.markdown("### HR Dashboard Actions")

    if st.session_state.hr_current_page == "hr_resume_ranking":
        hr_resume_ranking_app(user_id)
    elif st.session_state.hr_current_page == "hr_soft_skill_ranking":
        hr_soft_skill_ranking_app(user_id)
    elif st.session_state.hr_current_page == "hr_feedback":
        hr_feedback_app(user_id)
    elif st.session_state.hr_current_page == "hr_manage_history":
        hr_manage_history_app(user_id)
    elif st.session_state.hr_current_page == "hr_chatbot":
        hr_chatbot_app()
    elif st.session_state.hr_current_page is None:
        st.info("Welcome to the HR Professional Dashboard! Use the sidebar to navigate.")
        show_page("hr_resume_ranking")  # Set a default page



# --- Main App with Login and Role Selection ---
st.title("RankItRight")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "role" not in st.session_state:
    st.session_state["role"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "hr_current_page" not in st.session_state:
    st.session_state["hr_current_page"] = None
if "student_current_page" not in st.session_state:
    st.session_state["student_current_page"] = None



def show_hr_page(page_name):
    st.session_state.hr_current_page = page_name



def show_student_page(page_name):
    st.session_state.student_current_page = page_name



def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    roles = ["HR Professional", "Student"]
    selected_role = st.selectbox("Select your role", roles)

    if st.button("Login", use_container_width=True):
        if username and password:
            user_id, role = verify_user(username, password)
            if user_id:
                st.session_state["logged_in"] = True
                st.session_state["role"] = role
                st.session_state["user_id"] = user_id
                st.success(f"Logged in as {role}!")
                if role == "HR Professional":
                    st.session_state.hr_current_page = "hr_resume_ranking"  # Default HR page
                elif role == "Student":
                    st.session_state.student_current_page = "student_resume_checker"  # Default Student page
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Please enter username and password.")

    st.markdown("---")
    st.subheader("Create Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_role = st.selectbox("Select your role", roles, key="new_role")
    if st.button("Create Account", key="create_account_btn", use_container_width=True):
        if new_username and new_password:
            create_user(new_username, new_password, new_role)
        else:
            st.error("Please enter a username and password for the new account.")



if not st.session_state["logged_in"]:
    login_page()
else:
    if st.button("Logout", key="main_logout_btn", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["role"] = None
        st.session_state["user_id"] = None
        st.session_state.hr_current_page = None
        st.session_state.student_current_page = None
        st.rerun()

    if st.session_state["role"] == "HR Professional":
        hr_app(st.session_state["user_id"], show_hr_page)
    elif st.session_state["role"] == "Student":
        student_app(st.session_state["user_id"], show_student_page)