import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np

st.set_page_config(
    page_title="OCD SEVERITY PREDICTOR"
)

# Sidebar configuration
image_path = "./From-the-lecture/assets/OCD.jpeg"  # Adjust the path if needed

# Check if the image exists
if os.path.exists(image_path):
    st.sidebar.image(image_path)  # Display the image in the sidebar
else:
    st.sidebar.write("Image file not found, please check the path.")

# Navigation options in the sidebar (now with 4 options)
sidebar_option = st.sidebar.radio("Select an option", options=["About", "Predictor", "Descriptive Analytics", "Diagnostic Analytics"])

# "About" page for project description
if sidebar_option == "About":
    st.markdown("<h2 style='color: turquoise;'>About the Project</h2>", unsafe_allow_html=True)
    st.write("""
    
    This dashboard is designed to provide insights into the severity of OCD (Obsessive-Compulsive Disorder) symptoms according to the Y-BOCS (Yale-Brown Obsessive Compulsive Scale) scores. The Y-BOCS scores are used to assess the severity of obsessions and compulsions, to understand the impact of OCD (1).
    """)
    st.markdown("<h2 style='color: turquoise;'>About the Dataset</h2>", unsafe_allow_html=True)
    st.write("""

The dataset contains clinical information about individuals diagnosed with OCD. There were initially 17 features and 1500 patients but there were instances in the dataset that had total score above 40, and individual scores for Obsessions and Compulsions above 20 each, which is not acceptable based on the scoring system of the scale. Hence, we have eliminated those and a total of 412 patients were used in creating this prediction model. 

The scores for obsession and compulsion range from 0-20 each, and the total score ranges from 0-40. The dataset did not have the total score calculated which we have done by adding the scores for Y-BOCS Score (Obsessions) and Y-BOCS Score (Compulsions). The total score is categorized as:

0-20: Low , 
21-40: High. 
             
The dataset includes the following features:

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

# Page for predicting new patients (renamed to "Predictor")
elif sidebar_option == "Predictor":
    st.markdown(
        "<h1 style='color: turquoise;'>OCD PREDICTOR</h1>",
        unsafe_allow_html=True
    )

    # Input Form for user details
    st.markdown("<h2 style='color: turquoise;'>Patient Form</h2>", unsafe_allow_html=True)

    with st.form(key='patient_info_form'):
        # Input fields
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])  # Gender selection
        family_history = st.selectbox("Family History of OCD?", options=["Yes", "No"])  # Family history
        depression = st.selectbox("Does Patient have depression?", options=["Yes", "No"])  # Depression status
        anxiety = st.selectbox("Does Patient have anxiety?", options=["Yes", "No"])  # Anxiety status
        age = st.number_input("Age", min_value=0, max_value=120, value=25)  # Default age
        # Obsession and Compulsive type options
        obsession_type = st.selectbox("Obsession Type", options=["Harm-related", "Contamination", "Religious", "Hoarding", "Symmetry", "Else", "None"])
        compulsive_type = st.selectbox("Compulsive Type", options=["Checking", "Washing", "Counting", "Ordering", "Praying", "Else", "None"])
        # Duration of symptoms in whole years (integer input)
        symptom_duration_years = st.number_input("Duration of Symptoms (in years)", min_value=0, value=0, step=1, format="%d")  # Duration input in whole years

        submit_button = st.form_submit_button(label='Submit')

    # Display results only after the form is submitted
    if submit_button:
        st.success("Information Submitted Successfully!")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Family History of OCD:** {family_history}")
        st.write(f"**Depression:** {depression}")
        st.write(f"**Anxiety:** {anxiety}")
        st.write(f"**Age:** {age}")
        st.write(f"**Obsession Type:** {obsession_type}")
        st.write(f"**Compulsive Type:** {compulsive_type}")
        st.write(f"**Duration of Symptoms:** {symptom_duration_years} years")  # Display duration

        # Simplified severity prediction logic: Low or High Symptoms
        if symptom_duration_years < 1:  # Example condition for low symptoms
            predicted_severity = "Low Symptoms"
        else:
            predicted_severity = "High Symptoms"

        st.markdown(f"### Predicted: **{predicted_severity}**")

        # If "High Symptoms", show additional details about compulsive symptoms
        if predicted_severity == "High Symptoms":
            # Add a note about the type of symptoms (mostly compulsive)
            st.markdown("""
                #### Symptom Type: 
                - **Mostly Compulsive Symptoms**  
                (Compulsions are repetitive behaviors or mental acts that a person feels driven to perform.)
            """)

            # Optional: Add a description about what compulsive symptoms are
            st.markdown("""
            Compulsive symptoms often include actions like:
            - Repeatedly checking if the door is locked
            - Washing hands frequently
            - Counting items
            - Organizing things in a very specific order
            """)

            # Create and display an illustration of high symptoms
            def plot_high_symptoms_illustration():
                fig, ax = plt.subplots(figsize=(8, 4))

                # Set a gradient background
                ax.set_facecolor('#f2f2f2')  # Light gray background
                ax.set_xlim(0, 100)

                # Draw a red bar for high severity
                ax.barh(['High Severity'], [100], color='red', edgecolor='black', height=0.4)

                # Adding a title and centered text
                ax.set_title('High Severity Symptoms', fontsize=20, color='black', fontweight='bold', pad=20)
                ax.text(50, 0, 'High Symptoms', ha='center', va='center', fontsize=16, color='white', fontweight='bold')

                # Adding decorative elements
                for spine in ax.spines.values():
                    spine.set_visible(False)  # Hide spines for cleaner look

                # Remove y-ticks and x-ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Add a footer note
                plt.figtext(0.5, -0.1, 'This indicates high severity of symptoms, requiring attention.', 
                            wrap=True, horizontalalignment='center', fontsize=12, color='gray')

                return fig



            # Plot the high symptoms illustration and display it
            high_symptoms_fig = plot_high_symptoms_illustration()
            st.pyplot(high_symptoms_fig)
        elif predicted_severity == "Low Symptoms":
            # Add a note about the type of symptoms (mostly compulsive)
            st.markdown("""
                #### Symptom Type: 
                - **Mostly Obsessive Symptoms**  
            """)

            # Optional: Add a description about what compulsive symptoms are
            

            # Create and display an illustration of loe symptoms
            def plot_low_symptoms_illustration():
                fig, ax = plt.subplots(figsize=(8, 4))

                # Set a gradient background
                ax.set_facecolor('#f2f2f2')  # Light gray background
                ax.set_xlim(0, 100)

                # Draw a red bar for high severity
                ax.barh(['Low Severity'], [100], color='green', edgecolor='black', height=0.4)

                # Adding a title and centered text
                ax.set_title('Low Severity Symptoms', fontsize=20, color='black', fontweight='bold', pad=20)
                ax.text(50, 0, 'Low Symptoms', ha='center', va='center', fontsize=16, color='white', fontweight='bold')

                # Adding decorative elements
                for spine in ax.spines.values():
                    spine.set_visible(False)  # Hide spines for cleaner look

                # Remove y-ticks and x-ticks
                ax.set_xticks([])
                ax.set_yticks([])

                # Add a footer note
                plt.figtext(0.5, -0.1, 'This indicates Low severity of symptoms, not priortitized attention.', 
                            wrap=True, horizontalalignment='center', fontsize=12, color='gray')

                return fig

            # Plot the high symptoms illustration and display it
            low_symptoms_fig = plot_low_symptoms_illustration()
            st.pyplot(low_symptoms_fig)


# Placeholder for Descriptive Analytics
elif sidebar_option == "Descriptive Analytics":
    st.markdown("<h2 style='color: turquoise;'>Descriptive Analytics</h2>", unsafe_allow_html=True)
    st.write("This section will be filled with descriptive analytics.")

    # Loading the dataset 
    data = pd.read_csv('filtered_df.csv')
    # Display total number of patients
    num_patients = len(data)  
    st.metric("Total Number of Patients from the dataset used", num_patients)
    

    # Define the paths to the images
    image1_path = ".//From-the-lecture/assets/1a.jpg"
    image2_path = ".//From-the-lecture/assets/2a.jpg"
    image3_path = ".//From-the-lecture/assets/3a.jpg"
    image4_path = ".//From-the-lecture/assets/4a.jpg"
    image5_path = ".//From-the-lecture/assets/5a.jpg"
    image6_path = ".//From-the-lecture/assets/6a.jpg"
    image7_path = ".//From-the-lecture/assets/7a.jpg"
    image8_path = ".//From-the-lecture/assets/8a.jpg"
    image9_path = ".//From-the-lecture/assets/9a.jpg"
    image10_path = ".//From-the-lecture/assets/10a.jpg"
   
    st.image(image1_path, caption="Analytics Image 1", use_column_width=True)
    st.image(image2_path, caption="Analytics Image 2", use_column_width=True)
    st.image(image3_path, caption="Analytics Image 3", use_column_width=True)
    st.image(image4_path, caption="Analytics Image 4", use_column_width=True)
    st.image(image5_path, caption="Analytics Image 5", use_column_width=True)
    st.image(image6_path, caption="Analytics Image 6", use_column_width=True)
    st.image(image7_path, caption="Analytics Image 7", use_column_width=True)
    st.image(image8_path, caption="Analytics Image 8", use_column_width=True)
    st.image(image9_path, caption="Analytics Image 9", use_column_width=True)
    st.image(image10_path, caption="Analytics Image 10", use_column_width=True)

    
   

# Placeholder for Diagnostic Analytics
elif sidebar_option == "Diagnostic Analytics":
    st.markdown("<h2 style='color: turquoise;'>Diagnostic Analytics</h2>", unsafe_allow_html=True)
    st.write("This section will be filled with diagnostic analytics.")
