import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load AI model (runs once)
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------
# Job role skills database
# -------------------------------
job_roles = {
    "Java Developer": ["java", "spring", "hibernate", "mysql"],
    "Web Developer": ["html", "css", "javascript", "react"],
    "Python Developer": ["python", "django", "flask", "sql"]
}

# -------------------------------
# Learning roadmaps
# -------------------------------
learning_roadmaps = {
    "Java Developer": [
        "Learn Core Java",
        "Learn Spring Framework",
        "Learn Hibernate",
        "Practice MySQL",
        "Build REST API project"
    ],
    "Web Developer": [
        "Master HTML & CSS",
        "Learn JavaScript",
        "Learn React",
        "Learn Node.js",
        "Build full-stack project"
    ],
    "Python Developer": [
        "Strengthen Python basics",
        "Learn Flask/Django",
        "Practice SQL",
        "Work with APIs",
        "Build real-world project"
    ]
}

# -------------------------------
# Skill extraction (simple NLP)
# -------------------------------
def extract_skills(text):
    skills_db = [
        "python", "java", "html", "css", "javascript",
        "react", "django", "flask", "spring",
        "hibernate", "mysql", "sql"
    ]
    text = text.lower()
    found = [skill for skill in skills_db if skill in text]
    return list(set(found))

# -------------------------------
# Semantic matching (REAL AI)
# -------------------------------
def semantic_match(resume_text, job_text):
    resume_emb = model.encode([resume_text])
    job_emb = model.encode([job_text])
    score = cosine_similarity(resume_emb, job_emb)[0][0]
    return round(score * 100, 2)

# -------------------------------
# ATS Score
# -------------------------------
def calculate_ats_score(resume_text):
    score = 0
    feedback = []
    text = resume_text.lower()

    word_count = len(text.split())
    if 200 <= word_count <= 800:
        score += 20
    else:
        feedback.append("Optimize resume length (200–800 words).")

    if "skills" in text:
        score += 20
    else:
        feedback.append("Add a clear Skills section.")

    if "project" in text:
        score += 15
    else:
        feedback.append("Include Projects section.")

    if "experience" in text or "intern" in text:
        score += 20
    else:
        feedback.append("Add Experience or Internship.")

    action_verbs = ["developed", "built", "designed", "implemented", "created"]
    if any(word in text for word in action_verbs):
        score += 15
    else:
        feedback.append("Use strong action verbs.")

    if re.search(r"\d+%", text):
        score += 10
    else:
        feedback.append("Add measurable achievements (e.g., improved by 30%).")

    return score, feedback


# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🧠 AI-Based Resume Analyzer")
st.write("Analyze your resume, get job recommendations, and identify skill gaps using AI.")

resume_text = st.text_area(
    "📄 Paste your Resume Text",
    height=200,
    placeholder="Example:\nSkills: Python, Java, HTML\nProjects: Resume Analyzer using ML"
)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("🚀 Analyze Resume"):

    if not resume_text.strip():
        st.error("Please paste your resume text.")
        st.stop()

    # ✅ Extract skills
    detected_skills = extract_skills(resume_text)

    st.subheader("✅ Detected Skills")
    if detected_skills:
        st.success(", ".join(detected_skills))
    else:
        st.warning("No skills detected.")

    # -------------------------------
    # ATS Score
    # -------------------------------
    st.subheader("📊 ATS Resume Score")
    ats_score, ats_feedback = calculate_ats_score(resume_text)

    st.progress(ats_score / 100)
    st.write(f"**ATS Score:** {ats_score}/100")

    if ats_feedback:
        st.warning("### 🔧 Improvement Suggestions")
        for tip in ats_feedback:
            st.write(f"- {tip}")
    else:
        st.success("✅ Your resume looks ATS friendly!")

    # -------------------------------
    # Job Recommendations
    # -------------------------------
    st.subheader("📌 Job Recommendations & Skill Gap Analysis")

    for role, required_skills in job_roles.items():

        matched = list(set(detected_skills) & set(required_skills))
        missing = list(set(required_skills) - set(detected_skills))

        job_description = " ".join(required_skills)
        match_percentage = semantic_match(resume_text, job_description)

        st.markdown(f"### 🧩 {role}")
        st.write(f"**Match Percentage:** {match_percentage}%")
        st.write(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

        # ✅ Learning Roadmap
        if role in learning_roadmaps:
            st.info("🗺️ Learning Roadmap")
            for i, step in enumerate(learning_roadmaps[role], 1):
                st.write(f"{i}. {step}")

    st.divider()