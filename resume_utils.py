import pdfplumber
import pickle
import re


education_encoder = pickle.load(
    open(
        "education.pkl",
        "rb"
    )
)


# Extract text from PDF
def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

    return text




def analyze_resume(

        text,
        required_skills,
        min_experience,
        min_projects,
        required_education

):

    matched=[]
    missing=[]


    # Skill weights
    skill_weights={

        "Python":40,
        "SQL":25,
        "Machine Learning":25,
        "TensorFlow":10,
        "Java":35,
        "AWS":20

    }


    # Skill aliases
    skill_alias={

        "Machine Learning":[
            "machine learning",
            "ml"
        ],

        "Python":[
            "python"
        ],

        "Java":[
            "java"
        ],

        "SQL":[
            "sql",
            "mysql",
            "postgresql"
        ],

        "AWS":[
            "aws",
            "amazon web services"
        ],

        "TensorFlow":[
            "tensorflow",
            "tf"
        ]
    }


    skill_score=0


    for skill in required_skills:

        skill=skill.strip()

        found=False


        aliases=skill_alias.get(

            skill,

            [skill.lower()]

        )


        for alias in aliases:

            if alias.lower() in text.lower():

                found=True
                break


        if found:

            matched.append(skill)

            skill_score += skill_weights.get(
                skill,
                10
            )

        else:

            missing.append(skill)



    # Experience extraction

    exp=re.search(

        r'(\d+)\s*years',

        text.lower()

    )

    experience=int(

        exp.group(1)

    ) if exp else 0



    # Project extraction

    project_match=re.search(

        r'projects?\s*:?\s*(\d+)',

        text.lower()

    )

    projects=int(

        project_match.group(1)

    ) if project_match else 0



    # Education check

    if required_education=="Any Degree":

        education_ok=True

    else:

        education_ok=(

            required_education.lower()

            in text.lower()

        )



    # Score calculation

    score=skill_score


    if experience>=min_experience:

        score += 15


    if projects>=min_projects:

        score += 10


    if education_ok:

        score += 5



    score=min(score,100)



    result="Rejected"

    if score>=70:

        result="Selected"



    return {

    "score":score,

    "matched":matched,

    "missing":missing,

    "experience":experience,

    "projects":projects,

    "education_ok":education_ok,

    "experience_ok":experience>=min_experience,

    "projects_ok":projects>=min_projects,

    "skill_score":skill_score,

    "result":result

}