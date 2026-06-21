import re

def extract_jd_info(text):

    skills=[]

    skill_list=[

        "Python",
        "Java",
        "SQL",
        "Machine Learning",
        "TensorFlow",
        "AWS"

    ]

    for skill in skill_list:

        if skill.lower() in text.lower():

            skills.append(skill)


    exp=re.search(
        r'(\d+)\s*years',
        text.lower()
    )

    experience=int(
        exp.group(1)
    ) if exp else 0


    return {

        "skills":skills,
        "experience":experience
    }