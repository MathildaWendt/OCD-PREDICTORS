import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="About")

##Page information
st.markdown("<h2 style='color: turquoise;'>About the Project</h2>", unsafe_allow_html=True)
st.write("""
    
    This dashboard is designed to provide insights into the severity of OCD (Obsessive-Compulsive Disorder) symptoms according to the Y-BOCS (Yale-Brown Obsessive Compulsive Scale) scores. The Y-BOCS scores are used to assess the severity of obsessions and compulsions, to understand the impact of OCD (1).
    """)
st.markdown("<h2 style='color: turquoise;'>About the Dataset</h2>", unsafe_allow_html=True)
st.write("""

The dataset contains clinical information about individuals diagnosed with OCD. There were initially 17 features and 1500 patients but there were instances in the dataset that had total score above 40, and individual scores for Obsessions and Compulsions above 20 each, which is not acceptable based on the scoring system of the scale. Hence, we have eliminated those and now have a total of 412 patients were used in creating this prediction model. The dataset includes the following features:

1.	Patient ID: Identifier for each patient.
2.	Age: The patient's age at the time of the data collection.
3.	Gender: Gender identification of the patient.
4.	Ethnicity: The ethnic background of the patient.
5.	Marital Status: The patient's marital status.
6.	Education Level: The highest level of education completed by the patient.
7.	OCD Diagnosis Date: The date when the patient was diagnosed with OCD.
8.	Duration of Symptoms (months): The length of time (in months) the patient has experienced OCD symptoms.
9.	Previous Diagnoses: Any previous mental health diagnoses the patient has received.
10.	Family History of OCD: Whether the patient has a family history of OCD.
11.	Obsession Type: The specific types of obsessions the patient experiences.
12.	Compulsion Type: The specific types of compulsions the patient exhibits.
13.	Y-BOCS Score (Obsessions): The severity score of the patient's obsessions, as measured by the Y-BOCS.
14.	Y-BOCS Score (Compulsions): The severity score of the patient's compulsions, as measured by the Y-BOCS.
15.	Depression Diagnosis: Whether the patient has a comorbid diagnosis of depression.
16.	Anxiety Diagnosis: Whether the patient has a comorbid diagnosis of anxiety.
17.	Medications: Medications currently prescribed to the patient for managing OCD or related conditions.
""")
st.markdown("<h2 style='color: turquoise;'>Group Members</h2>", unsafe_allow_html=True)
st.write("""

1.	Amanda Bashiri
2.	Mathilda Wendt
3.	Moa Osterberg
4.	Shweta Prasad Ghaisas
5.	Sidra Jahanzeb
6.	Ye Htut
""")
st.markdown("<h2 style='color: turquoise;'>Contact Information</h2>", unsafe_allow_html=True)
st.write("""

sid.amjed87@gmail.com
""")
st.markdown("<h2 style='color: turquoise;'>Reference</h2>", unsafe_allow_html=True)
st.write("""

1.	Bejerot S, Edman G, Anckarsäter H, Berglund G, Gillberg C, Hofvander B, et al. The Brief Obsessive-Compulsive Scale (BOCS): a self-report scale for OCD and obsessive-compulsive related disorders. Nord J Psychiatry. 2014 Nov;68(8):549-59. doi: 10.3109/08039488.2014.884631.

    """)
