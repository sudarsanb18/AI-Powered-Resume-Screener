import streamlit as st
import matplotlib.pyplot as plt

from resume_utils import extract_text, analyze_resume
from ranking import rank_candidates
from dashboard import dashboard_data


st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)


st.title("AI Resume Screening System")

st.header("HR Requirements")


skills = st.text_input(
    "Required Skills",
    "Python,SQL,Machine Learning"
)


col1,col2 = st.columns(2)

with col1:

    min_exp = st.number_input(
        "Minimum Experience",
        value=2
    )

with col2:

    min_projects = st.number_input(
        "Minimum Projects",
        value=2
    )


education = st.selectbox(
    "Required Education",
    [

        "Any Degree",

        "B.Tech",
        "M.Tech",

        "B.E",
        "M.E",

        "B.Sc",
        "M.Sc",

        "BCA",
        "MCA",

        "B.Com",
        "M.Com",

        "BBA",
        "MBA",

        "BA",
        "MA",

        "PhD"

    ]
)



uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)



if uploaded_files:

    candidate_results=[]


    for file in uploaded_files:

        with open(
            file.name,
            "wb"
        ) as f:

            f.write(
                file.getbuffer()
            )


        text=extract_text(
            file.name
        )


        result=analyze_resume(

            text,
            skills.split(","),
            min_exp,
            min_projects,
            education

        )


        st.divider()

        st.subheader(
            f"📄 {file.name}"
        )


        st.metric(
            "Eligibility Score",
            f"{result['score']}%"
        )

        st.progress(
            result["score"]/100
        )


        st.write(
            "## Requirement Scoreboard"
        )


        st.write(
            "### Skills Match"
        )

        for skill in skills.split(","):

            skill=skill.strip()

            if skill in result["matched"]:

                st.success(
                    f"✔ {skill}"
                )

            else:

                st.error(
                    f"✘ {skill}"
                )



        st.write(
            "### Experience"
        )

        if result["experience_ok"]:

            st.success(

                f"✔ Required: {min_exp} years | "
                f"Candidate: {result['experience']} years"

            )

        else:

            st.error(

                f"✘ Required: {min_exp} years | "
                f"Candidate: {result['experience']} years"

            )



        st.write(
            "### Projects"
        )

        if result["projects_ok"]:

            st.success(

                f"✔ Required: {min_projects} | "
                f"Candidate: {result['projects']}"

            )

        else:

            st.error(

                f"✘ Required: {min_projects} | "
                f"Candidate: {result['projects']}"

            )



        st.write(
            "### Education"
        )

        if result["education_ok"]:

            st.success(
                f"✔ {education} matched"
            )

        else:

            st.error(
                f"✘ {education} not found"
            )



        if result["result"]=="Selected":

            st.success(
                "Candidate Selected"
            )

        else:

            st.error(
                "Candidate Rejected"
            )



        candidate_results.append({

            "name":file.name,
            "score":result["score"],
            "result":result["result"]

        })



    ranked=rank_candidates(
        candidate_results
    )


    st.divider()


    st.subheader(
        "🏆 Top Candidate"
    )

    top=ranked[0]


    st.success(

        f"{top['name']} "
        f"({top['score']}%)"

    )



    stats=dashboard_data(
        candidate_results
    )


    st.subheader(
        "Dashboard"
    )


    c1,c2,c3=st.columns(3)


    with c1:

        st.metric(
            "Total Candidates",
            stats["total"]
        )


    with c2:

        st.metric(
            "Selected",
            stats["selected"]
        )


    with c3:

        st.metric(
            "Rejected",
            stats["rejected"]
        )



    st.subheader(
        "Candidate Ranking"
    )


    medals=["🥇","🥈","🥉"]


    for i,candidate in enumerate(ranked):

        icon="📄"

        if i<3:

            icon=medals[i]


        st.write(

            f"{icon} "
            f"{candidate['name']} "
            f"- {candidate['score']}%"
        )


        st.progress(
            candidate["score"]/100
        )



    st.subheader(
        "Candidate Eligibility Dashboard"
    )


    names=[]
    scores=[]


    for candidate in ranked:

        names.append(
            candidate["name"]
        )

        scores.append(
            candidate["score"]
        )


    fig,ax=plt.subplots(
        figsize=(10,5)
    )


    ax.barh(
        names,
        scores
    )

    ax.set_xlabel(
        "Eligibility Score (%)"
    )

    ax.set_title(
        "Candidate Ranking Analysis"
    )


    st.pyplot(fig)