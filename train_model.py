import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("resume_data.csv")

df.drop(
    ['Resume_ID','Name'],
    axis=1,
    inplace=True
)

df['Skills_Count'] = df['Skills'].apply(
    lambda x: len(str(x).split(","))
)

df['Certifications_Count']=df[
    'Certifications'
].apply(
    lambda x: len(str(x).split(","))
)

education_encoder=LabelEncoder()

df['Education']=education_encoder.fit_transform(
    df['Education']
)

decision_encoder=LabelEncoder()

df['Recruiter Decision']=decision_encoder.fit_transform(
    df['Recruiter Decision']
)

X=df[[
    'Skills_Count',
    'Experience (Years)',
    'Education',
    'Certifications_Count',
    'Projects Count'
]]

y=df['Recruiter Decision']

model=RandomForestClassifier()

model.fit(X,y)

pickle.dump(
    model,
    open("model.pkl","wb")
)

pickle.dump(
    education_encoder,
    open("education.pkl","wb")
)

print(
    "Model trained successfully"
)