import streamlit as st
import pandas as pd

import pickle
import os



@st.cache(show_spinner=False, allow_output_mutation=True)
def load_models():
    xgboost = pickle.load(open(os.path.join(os.getcwd(), "model.pkl"), "rb"))

    return xgboost  


def predict(input_dataframe: pd.DataFrame):
    print(input_dataframe)
    input_dataframe = input_dataframe.drop(
        columns=["CholCheck", "Stroke", "HeartDiseaseorAttack", 
                   'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost'])

    y_test_predicted_cat = xgboost.predict(input_dataframe)

    print(y_test_predicted_cat)

    with st.container():
        st.header("Output")
        if len(y_test_predicted_cat) == 1:
            output = 'Diabetic' if y_test_predicted_cat[0] == 1 else 'Non-Diabetic'
            st.write(f"The predicted output is '{output}'")



st.title('Machine Learning On Diabetes Health Indicators Dataset')

with st.spinner("Loading Model"):
    xgboost = load_models()

st.write("This is the app for predicting whether a person is diabetic or not depending upon the input features.")
st.caption("This project is done with love by FMA-group fro UJ university, Data Science section.\n
Team members: Faisal Alzahrani, Mohammed Al-khathami, Anas  Al-Salami\n")
st.header('Please answare the following questions')


input_dataframe = None


st.caption("If you needed any help, see the '?' sign above each question.")
HighBP = st.number_input('Do you have a high blood preasure?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
HighChol = st.number_input('Do you have a high cholesterol?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
CholCheck = st.number_input('Did you check your cholestrol in the past 5 years?', value=1, min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
Bmi = st.number_input('What is your BMI?', value=40,help="Please enter a correct number")
Smoker = st.number_input('Are you somking?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
Stroke = st.number_input('Did you ever told that you had a stroke?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
HeartDiseaseorAttack = st.number_input('Do you have a heart disease/attack?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
PhysActivity = st.number_input('Were you physically active in the past 30 days?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
Fruits = st.number_input('Do you consume fruits one time or more per day?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
Veggies = st.number_input('Do you consume vegitables one time or more per day?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
HvyAlcoholConsump = st.number_input('Are you heavy alchohol drinker?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'(also if you do not drink)")
AnyHealthcare = st.number_input('Do you have any health care coverage?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
NoDocbcCost = st.number_input('Did you ever needed to see a doctor but could not because of the cost?',value= 0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
GenHlth = st.number_input('On scale of 1-5, how is your general health?(physically or mentally)',value= 5,min_value=1,max_value=5,step=1,help="Scale: 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor")
MentHlth = st.number_input('On scale of 0-30, how many days you felt good about your mental health in the past 30 days?', value=18,min_value=0,max_value=30,step=1,help="Scale: 0-30")
PhysHlth = st.number_input('On scale of 0-30, how many days you felt good about your physical health in the past 30 days?', value=15,min_value=0,max_value=30,step=1,help="Scale: 0-30")
DiffWalk = st.number_input('Do you have any walking difficulty?', value=1,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Yes' and 0 for 'No'")
Sex = st.number_input('What is your gender?', value=0,min_value=0,max_value=1,step=1,help="Please enter 1 for 'Male' and 0 for 'Female'")
Age = st.number_input('In what category is your age? (1 = 18-24 ,9 = 60-64 ,13 = 80 or older)',value= 9,min_value=1,max_value=13,step=1,help='''1 Age 18 to 24\n
2 Age 25 to 29\n
3 Age 30 to 34\n
4 Age 35 to 39\n
5 Age 40 to 44\n
6 Age 45 to 49\n
7 Age 50 to 54\n
8 Age 55 to 59\n
9 Age 60 to 64\n
10 Age 65 to 69\n
11 Age 70 to 74\n
12 Age 75 to 79\n
13 Age 80 or older''')
Education = st.number_input('On scale of 1-6, in what level is your education?', value=4,min_value=1,max_value=6,step=1,help='''1 = Never attended school or only kindergarten \n
2 = Grades 1 through 8 (Elementary) \n
3 = Grades 9 through 11 (Some high school) \n
4 = Grade 12 or GED (High school graduate) \n
5 = College 1 year to 3 years (Some college or technical school) \n
6 = College 4 years or more (College graduate)\n
Scale:1-6''')
Income = st.number_input('On scale of 1-8, in what level is you income?',value= 3,min_value=1,max_value=8,step=1,help='''1= Less than $10,000
2= Less than $15,000 ($10,000 to less than $15,000)\n
3= Less than $20,000 ($15,000 to less than $20,000)\n
4= Less than $25,000 ($20,000 to less than $25,000)\n
5= Less than $35,000 ($25,000 to less than $35,000)\n
6= Less than $50,000 ($35,000 to less than $50,000)\n
7= Less than $75,000 ($50,000 to less than $75,000)\n
8= $75,000 or more''')

l = [[HighBP, HighChol,CholCheck, Bmi, Smoker,Stroke,HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
        HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education,Income]]

input_dataframe = pd.DataFrame(l, columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
'Income'])



submit = st.button('Submit')

if submit:
    predict(input_dataframe)
