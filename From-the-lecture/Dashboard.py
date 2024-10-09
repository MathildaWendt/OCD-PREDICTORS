import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(
    page_title="OCD SEVERITY PREDICTOR"
)

# Sidebar configuration
image_path = ".//From-the-lecture/assets/OCD.jpeg"  # Adjust the path if needed

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
    Description about the project 

    **Future work**
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

# Placeholder for Descriptive Analytics
elif sidebar_option == "Descriptive Analytics":
    st.markdown("<h2 style='color: turquoise;'>Descriptive Analytics</h2>", unsafe_allow_html=True)
    st.write("This section will be filled with descriptive analytics.")
    

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
