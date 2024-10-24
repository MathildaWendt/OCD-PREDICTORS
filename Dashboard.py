import streamlit as st
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import chi2_contingency
import seaborn as sns
import joblib
import shap 



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
sidebar_option = st.sidebar.radio("Select an option", options=["About", "Predictive Analytics", "Descriptive Analytics", "Diagnostic Analytics"])

# "About" page for project description
if sidebar_option == "About":
    st.markdown("<h2 style='color: #007D79;'>Welcome to OCD Symptom Severity Prediction Dashboard</h2>", unsafe_allow_html=True)

    
    st.markdown("<h3 style='color: turquoise;'>About the Project</h3>", unsafe_allow_html=True)
    st.write("""
    
    This dashboard is designed to provide insights into the severity of OCD (Obsessive-Compulsive Disorder) symptoms according to the Y-BOCS (Yale-Brown Obsessive Compulsive Scale) scores. The Y-BOCS scores are used to assess the severity of obsessions and compulsions, to understand the impact of OCD (1).
    """)
    st.markdown("<h3 style='color: turquoise;'>About the Dataset</h3>", unsafe_allow_html=True)
    st.write("""

The dataset contains clinical information about individuals diagnosed with OCD. There were initially 17 features and 1500 patients but there were instances in the dataset that had total score above 40, and individual scores for Obsessions and Compulsions above 20 each, which is not acceptable based on the scoring system of the scale. Hence, we have eliminated those and a total of 412 patients were used in creating this prediction model. 
             
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

We were interested in predicting the severity of OCD given by "Total_Score", therefore we combined features 13 and 14. Based on a literature review found that age, duration of symptoms, family history of OCD, and diagnosis of Anxiety or Depression affect it. Gender being a basic demographic feature has been retained though research does not clearly indicate its influence on OCD. Obsession Type and Compulsion Type are a reflection of patient's symptoms which may affect severity of the disorder, and therefore have been retained (3,4,5,6,7). 

The scores for obsession and compulsion range from 0-20 each, and the total score ranges from 0-40. The dataset did not have the total score calculated which we have done by adding the scores for Y-BOCS Score (Obsessions) and Y-BOCS Score (Compulsions). The total score is categorized as:

0-20: Low , 
21-40: High.

""")
    st.markdown("<h3 style='color: turquoise;'>Group Members</h3>", unsafe_allow_html=True)
    st.write("""

1.	Amanda Bashiri
2.	Mathilda Wendt
3.	Moa Osterberg
4.	Shweta Prasad Ghaisas
5.	Sidra Jahanzeb
6.	Ye Htut
""")
    st.markdown("<h3 style='color: turquoise;'>Contact Information</h3>", unsafe_allow_html=True)
    st.write("""

1. amanda.b-14@hotmail.com
2. mathilda.hedbeck@gmail.com            
3. moaosterberg@gmail.com
4. shwetaghaisas@gmail.com
5. sid.amjed87@gmail.com
6. dr.yehtut.1995@gmail.com
""")
    st.markdown("<h5 style='color: turquoise;'>Reference</h5>", unsafe_allow_html=True)
    st.write("""

1.	Bejerot S, Edman G, Anckarsäter H, Berglund G, Gillberg C, Hofvander B, et al. The Brief Obsessive-Compulsive Scale (BOCS): a self-report scale for OCD and obsessive-compulsive related disorders. Nord J Psychiatry. 2014 Nov;68(8):549-59. doi: 10.3109/08039488.2014.884631.
2. Haque O, Alamgir Z. OCD Patient Dataset: Demographics and Clinical Data [Internet].2023 [cited 2024-09-08]. Available from: https://www.kaggle.com/datasets/ohinhaque/ocd-patient-dataset-demographics-and-clinical-data/data
3. Mathes BM, Morabito DM, Schmidt NB. Epidemiological and Clinical Gender Differences in OCD. Curr Psychiatry Rep. 2019 Apr 23;21(5):36. doi: 10.1007/s11920-019-1015-2.
4. Riddle DB, Guzick A, Minhajuddin A, Smárason O, Armstrong GM, Slater H, et al. Obsessive-compulsive disorder in youth and young adults with depression: Clinical characteristics of comorbid presentations. J Obsessive Compuls Relat Disord. 2023 Jul;38:100820. doi: 10.1016/j.jocrd.2023.100820. 
5. Zheng H, Zhang Z, Huang C, Luo G. Medical status of outpatients with obsessive-compulsive disorder in psychiatric department and its influencing factors. Zhong Nan Da Xue Xue Bao Yi Xue Ban. 2022 Oct 28;47(10):1418-1424. English, Chinese. doi: 10.11817/j.issn.1672-7347.2022.220125. 
6. Mahjani B, Bey K, Boberg J, Burton C. Genetics of obsessive-compulsive disorder. Psychol Med. 2021 Oct;51(13):2247-2259. doi: 10.1017/S0033291721001744. Epub 2021 May 25. PMID: 34030745; PMCID: PMC8477226.
7. Mathes BM, Morabito DM, Schmidt NB. Epidemiological and Clinical Gender Differences in OCD. Curr Psychiatry Rep. 2019 Apr 23;21(5):36. doi: 10.1007/s11920-019-1015-2. PMID: 31016410.

""")

# Placeholder for Predictive Analytics
elif sidebar_option == "Predictive Analytics":
    st.markdown("<h2 style='color: turquoise;'>Predictive Analytics</h2>", unsafe_allow_html=True)
    st.write("On this tab, please fill in the demographic and clinical details of the patient for whom you wish to find out the predicted severity of OCD. Hit 'Submit'. The algorithm predicts whether the severity of OCD in this patient is 'High' or 'Low'. Following the prediction, you will see an explanation of the prediction.")

# Load the trained model (Random Forest)
    rf_model = joblib.load('rf_model.pkl')

# Load the dataset to obtain scaling parameters
    new_df = pd.read_csv('output_file.csv')

    feature_columns=list(new_df.columns)[:-1] #exclude last column which is the label

# Create form for user input
    with st.form(key='prediction_form'):
    # Input fields
        age = st.number_input("Age", min_value=18, max_value=120, value=25)
        family_history = st.selectbox("Family History of OCD?", options=["Yes", "No"])
        symptom_duration_months = st.number_input("Duration of Symptoms (in months)", min_value=0, value=0, step=1, format="%d")
        depression = st.selectbox("Does Patient have depression?", options=["Yes", "No"])
        anxiety = st.selectbox("Does Patient have anxiety?", options=["Yes", "No"])
        gender = st.selectbox("Gender", options=["Male", "Female"])

    # Obsession and Compulsion type options
        obsession_type = st.selectbox("Obsession Type", options=["Harm-related", "Contamination", "Religious", "Hoarding", "Symmetry"])
        compulsive_type = st.selectbox("Compulsion Type", options=["Checking", "Washing", "Counting", "Ordering", "Praying"])

    # Submit button
        submit_button = st.form_submit_button(label='Submit')

# Process user input if submit button is clicked
    if submit_button:
    # Convert inputs to match the format of new_df
        user_data = {
            'Age': age,
            'Family History of OCD': 1 if family_history == 'Yes' else 0,
            'Duration of Symptoms (months)': symptom_duration_months,
            'Depression Diagnosis': 1 if depression == 'Yes' else 0,
            'Anxiety Diagnosis': 1 if anxiety == 'Yes' else 0,
            'Gender_Boolean': 1 if gender == 'Male' else 0,
            'Obsession Type_Contamination': 1 if obsession_type == 'Contamination' else 0,
            'Obsession Type_Harm-related': 1 if obsession_type == 'Harm-related' else 0,
            'Obsession Type_Hoarding': 1 if obsession_type == 'Hoarding' else 0,
            'Obsession Type_Religious': 1 if obsession_type == 'Religious' else 0,
            'Obsession Type_Symmetry': 1 if obsession_type == 'Symmetry' else 0,
            'Compulsion Type_Checking': 1 if compulsive_type == 'Checking' else 0,
            'Compulsion Type_Counting': 1 if compulsive_type == 'Counting' else 0,
            'Compulsion Type_Ordering': 1 if compulsive_type == 'Ordering' else 0,
            'Compulsion Type_Praying': 1 if compulsive_type == 'Praying' else 0,
            'Compulsion Type_Washing': 1 if compulsive_type == 'Washing' else 0
        }

    # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data], columns=feature_columns)

    # Load scaler used for "Age" and "Duration of Symptoms (months)"
        scaler = joblib.load('scaler.pkl')

    # Normalize user data using the same scaler
        normalized_features = scaler.transform(user_df[["Age", "Duration of Symptoms (months)"]])
        user_df["Age"].iloc[0]=normalized_features[0,0]
        user_df["Duration of Symptoms (months)"].iloc[0]=normalized_features[0,1]

    # Make prediction using the loaded Random Forest model
        prediction = rf_model.predict(user_df)

        
    # Gives back the input. 
        st.success("Information Submitted Successfully!")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Family History of OCD:** {family_history}")
        st.write(f"**Depression:** {depression}")
        st.write(f"**Anxiety:** {anxiety}")
        st.write(f"**Age:** {age}")
        st.write(f"**Obsession Type:** {obsession_type}")
        st.write(f"**Compulsive Type:** {compulsive_type}")
        st.write(f"**Duration of Symptoms:** {symptom_duration_months} months

        st.markdown("### Predicted Severity of OCD:")
    # Display the prediction result
        prediction_label = 'Low' if prediction == 0 else 'High'
        if prediction_label == 'High':
        # Display red box for High severity prediction
            st.markdown(
                """
                <div style="background-color: red; padding: 20px; border-radius: 10px;">
                    <h2 style="color: white; text-align: center;">High</h2>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(
                """
                <p style="text-align: center; color: black; font-size: 16px;">
                This prediction indicates high severity of symptoms. Requires attention.
                </p>
                """, unsafe_allow_html=True
            )
        else:
        # Display green box for Low severity prediction
            st.markdown(
                """
                <div style="background-color: green; padding: 20px; border-radius: 10px;">
                    <h2 style="color: white; text-align: center;">Low</h2>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(
               """
               <p style="text-align: center; color: black; font-size: 16px;">
               This prediction indicates low severity of symptoms. Attention not prioritized.
               </p>
               """, unsafe_allow_html=True
            )

# SHAP Explanation Section
        st.markdown("<h2 style='color: turquoise;'>Prescriptive Analytics</h2>", unsafe_allow_html=True)
        st.write("This section helps you understand which patient factors, like demographics and clinical details, influenced the algorithm's prediction of OCD severity. Prescriptive analytics can guide you in setting treatment goals for the patient. The analytics presented here are specific to the patient for whom you entered the above data.")
        st.write("We have used SHAP (SHapley Additive exPlanations) to explain the output of machine learning model. It provides insights into how much each feature contributed to a specific prediction, making complex models more interpretable.")

    # Create an expander for the SHAP Waterfall Plot
        with st.expander("SHAP Waterfall Plot", expanded=False):

    # Initialize the SHAP explainer
            explainer = shap.TreeExplainer(rf_model)
    
    # Compute SHAP values for the user input
            shap_values = explainer.shap_values(user_df)

    # Create an Explanation object for the SHAP values
            shap_explanation = shap.Explanation(
            values=shap_values[0, :, int(prediction)], 
            base_values=explainer.expected_value[int(prediction)], 
        #data=user_df.values[0], 
            feature_names=user_df.columns
            )
    
    # Use Matplotlib to render the plot in Streamlit
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_explanation)  # No ax argument needed
            st.pyplot(fig)

    # Add explanatory text
            st.markdown("""
            *Interpreting the SHAP Waterfall Plot*            
            *Features:* The features are listed along the vertical axis. The top features are listed in descending order of their importance, showing which features had the greatest impact on the specific prediction.
            *SHAP Values:* The horizontal bars represent the SHAP values, indicating the impact of each feature on the prediction. Positive values push the prediction towards the predicted class, while negative values push it away from the prediction.  
            *Base Value:* The base value, shown as a dashed line E[f(x)], represents the average model output across the training dataset. The sum of the SHAP values, when added to the base value, gives the final model output for that specific prediction denoted as f(x).
            """)

     # Detailed insights based on SHAP values
            top_features = pd.Series(shap_explanation.values, index=shap_explanation.feature_names).sort_values(ascending=False)       # Displaying top features with explanations
            for feature, value in top_features.items():
                impact = "positive" if value > 0 else "negative"
                st.write(f"- *{feature}:* {value:.3f} (This feature has a {impact} impact on the prediction.)")


# Placeholder for Descriptive Analytics
elif sidebar_option == "Descriptive Analytics":
    st.markdown("<h1 style='color: turquoise;'>Descriptive Analytics</h1>", unsafe_allow_html=True)

    # Loading the dataset 
    data = pd.read_csv('filtered_df.csv')
    ####### Patients, age, female, male####

    
    average_age = round(data['Age'].mean())
    num_patients = len(data)
    num_males = data['Gender'].value_counts().get('Male', 0)
    num_females = data['Gender'].value_counts().get('Female', 0)


    col1, col2, col3, col4 = st.columns(4)


    def create_stat_circle(value, label, emoji):
       return f"""
       <div style='display: flex; flex-direction: column; align-items: center;'>
            <div style='border: 2px solid #D3D3D3; border-radius: 50%; width: 100px; height: 100px; display: flex; 
                       align-items: center; justify-content: center; font-size: 24px; font-weight: bold; color: turquoise;'>
                {value}
            </div>
            <p style='text-align: center; margin-top: 5px; font-size: 16px;'>{emoji} {label}</p>
         </div>
      """

# patients
    with col1:
        st.markdown(create_stat_circle(num_patients, "Total Patients", "🧑‍🤝‍🧑"), unsafe_allow_html=True)

# age
    with col2:
        st.markdown(create_stat_circle(round(average_age, 2), "Average Age", "🎂"), unsafe_allow_html=True)

# men
    with col3:
        st.markdown(create_stat_circle(num_males, "Males", "👨"), unsafe_allow_html=True)

# females
    with col4:
        st.markdown(create_stat_circle(num_females, "Females", "👩"), unsafe_allow_html=True)

#####################################################################################

   ##
     # Age and Gender Distribution
    
    st.markdown("""
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 5px; border: 2px solid #D3D3D3;">
            <h3 style="color: turquoise;">Age and Gender Distribution</h3>
    """, unsafe_allow_html=True)

    # Creating age groups (bins)
    data['Age Group'] = pd.cut(data['Age'], bins=[0, 18, 30, 50, 70, 100], 
                                labels=['0-18', '19-30', '31-50', '51-70', '71+'])

    # Count the number of males and females in each age group
    gender_age_group = data.groupby(['Age Group', 'Gender']).size().unstack().fillna(0)

    # Create a figure
    fig0 = go.Figure()

    # Add bar plot for males
    fig0.add_trace(
        go.Bar(
            x=gender_age_group.index,
            y=gender_age_group['Male'],
            name='Male',
            marker_color='#AFEEEE'
        )
    )

    # Add bar plot for females
    fig0.add_trace(
        go.Bar(
            x=gender_age_group.index,
            y=gender_age_group['Female'],
            name='Female',
            marker_color='#E6E6FA'
        )
    )

    # Update layout to create stacked bars
    fig0.update_layout(
        barmode='stack',  # Stacked bar mode
        title='Number of People by Age Group and Gender',
        xaxis_title='Age Group',
        yaxis_title='Number of People',
        legend_title='Gender',
        xaxis_tickangle=0,
        yaxis=dict(showgrid=True)
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig0)

    
#########################################################################################

    st.markdown("""
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 5px; border: 2px solid #D3D3D3;">
            <h3 style="color: turquoise;">Diagnosis in Relation to Total Score</h3>
    """, unsafe_allow_html=True)
    

# Filters
    st.markdown("""
    This chart explores the relationship between **family history of OCD** and the **distribution of Total OCD Severity Scores**. 
    Additionally, you can use the **dropdown menus above** to filter the data based on whether the patient has been diagnosed with **depression** or **anxiety**. Adjusting these filters will allow you to visualize how these conditions may effect OCD severity.
    """)

# Create three columns for filters
    col1, col2, col3 = st.columns(3)

# Filter for Anxiety Diagnosis
    with col1:
        anxiety_options = [ 'Yes', 'No']
        selected_anxiety = st.selectbox("Anxiety Diagnosis", anxiety_options, key="anxiety_filter")

# Filter for Depression Diagnosis
    with col2:
        depression_options = [ 'Yes', 'No']
        selected_depression = st.selectbox("Depression Diagnosis", depression_options, key="depression_filter")

# Filter for Family History of OCD
    with col3:
        family_history_options = ['All', 'Yes', 'No']
        selected_family_history = st.selectbox("Family History of OCD", family_history_options, key="family_history_filter")

# Create a copy of the original data for filtering
    filtered_data = data.copy()

# Apply filters based on selections
    if selected_anxiety == 'Yes':
        filtered_data = filtered_data[filtered_data['Anxiety Diagnosis'] == True]
    elif selected_anxiety == 'No':
        filtered_data = filtered_data[filtered_data['Anxiety Diagnosis'] == False]

    if selected_depression == 'Yes':
        filtered_data = filtered_data[filtered_data['Depression Diagnosis'] == True]
    elif selected_depression == 'No':
        filtered_data = filtered_data[filtered_data['Depression Diagnosis'] == False]

    if selected_family_history == 'Yes':
        filtered_data = filtered_data[filtered_data['Family History of OCD'] == True]
    elif selected_family_history == 'No':
        filtered_data = filtered_data[filtered_data['Family History of OCD'] == False]

# Check if the filtered data is empty
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
    # Count the number of people with and without family history of OCD
        family_history_counts = filtered_data['Family History of OCD'].value_counts()

    # Create a bar plot for the number of people with and without family history of OCD
        bar_trace = go.Bar(
            x=family_history_counts.index, 
            y=family_history_counts.values, 
            name='Number of People',
            yaxis='y1',
            marker_color='#AFEEEE'
        )

    # Create a box plot for the distribution of Total_Score for each category
        box_trace = go.Box(
            x=filtered_data['Family History of OCD'], 
            y=filtered_data['Total_Score'], 
            name='Distribution of Total Score',
            yaxis='y2',
            marker_color='#9370DB'
        )

    # Combine the bar and box plot using secondary y-axes
        fig4 = go.Figure(data=[bar_trace, box_trace])

    # Update layout for dual y-axes
        fig4.update_layout(
            title='Number of People with Family History of OCD in relation to the Total Score',
            xaxis_title='Family History of OCD',
            yaxis=dict(
                title='Number of People',
                showgrid=False
            ),
            yaxis2=dict(
                title='Distribution of Total Score',
                overlaying='y',  # Overlay on the same plot
                side='right'
            ),
            legend=dict(x=0.1, y=1.1)
        )

    # Show the figure
        st.plotly_chart(fig4)


    #############################################################################################3

    st.markdown("""
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 5px; border: 2px solid #D3D3D3;">
            <h3 style="color: turquoise;">Analysis of Type of Symptom with Total Score</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This chart explores the relationship between the **selected symptom type** (either Obsession or Compulsion) and the **Average Total Score**.
    
    Use the **dropdown menu** to choose between **Obsession Types** and **Compulsion Types**, and apply filters to customize your analysis.
    """)

        # Selectbox for the type of analysis (Obsession or Compulsion)
    analysis_type = st.selectbox("Choose the type of Symptom to analyze:", ["Obsession Type", "Compulsion Type"])

    if analysis_type == "Obsession Type":
        # Multiselect for filtering by specific Obsession Types
        selected_obsessions = st.multiselect(
            "Select Obsession Types to Filter", 
            options=data['Obsession Type'].unique(),
            default=data['Obsession Type'].unique()  # Default is to select all
        )

        # Filter the dataset based on the selected Obsession Types
        filtered_data = data[data['Obsession Type'].isin(selected_obsessions)]

        # Count and Average Score for the selected obsession types
        obsession_counts = filtered_data['Obsession Type'].value_counts()
        obsession_avg_scores = filtered_data.groupby('Obsession Type')['Total_Score'].mean()

        # Create a figure for Obsession Type
        fig = go.Figure()

        # Add bar plot for the number of instances
        fig.add_trace(
            go.Bar(
                x=obsession_counts.index,
                y=obsession_counts.values,
                name='Number of Instances',
                marker_color='#AFEEEE',
                text=obsession_counts.values,  # Show count on the bars
                textposition='auto'
            )
        )

        # Add line plot for the average total score
        fig.add_trace(
            go.Scatter(
                x=obsession_avg_scores.index,
                y=obsession_avg_scores.values,
                name='Average Total Score',
                yaxis='y2',  # Associate this with the second y-axis
                mode='lines+markers',
                line=dict(color='#9370DB', width=3),
                marker=dict(size=8),
                text=obsession_avg_scores.values,  # Show average scores on the points
                textposition='top center'
            )
        )

        # Update layout to include a second y-axis
        fig.update_layout(
            title='Number of Instances and Average Total Score by Obsession Type',
            xaxis_title='Obsession Type',
            yaxis_title='Number of Instances',
            yaxis2=dict(
                title='Average Total Score',
                overlaying='y',  # Overlay y-axis 2 on the same plot as y-axis 1
                side='right'
            ),
            legend=dict(x=0.1, y=1.1),
            xaxis_tickangle=45,
            yaxis=dict(showgrid=True)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    elif analysis_type == "Compulsion Type":
        # Multiselect for filtering by specific Compulsion Types
        selected_compulsions = st.multiselect(
            "Select Compulsion Types to Filter", 
            options=data['Compulsion Type'].unique(),
            default=data['Compulsion Type'].unique()  # Default is to select all
        )

        # Filter the dataset based on the selected Compulsion Types
        filtered_data = data[data['Compulsion Type'].isin(selected_compulsions)]

        # Count and Average Score for the selected compulsion types
        compulsion_counts = filtered_data['Compulsion Type'].value_counts()
        compulsion_avg_scores = filtered_data.groupby('Compulsion Type')['Total_Score'].mean()

        # Create a figure for Compulsion Type
        fig = go.Figure()

        # Add bar plot for the number of instances
        fig.add_trace(
            go.Bar(
                x=compulsion_counts.index,
                y=compulsion_counts.values,
                name='Number of Instances',
                marker_color='#AFEEEE',
                text=compulsion_counts.values,  # Show count on the bars
                textposition='auto'
            )
        )

        # Add line plot for the average total score
        fig.add_trace(
            go.Scatter(
                x=compulsion_avg_scores.index,
                y=compulsion_avg_scores.values,
                name='Average Total Score',
                yaxis='y2',  # Associate this with the second y-axis
                mode='lines+markers',
                line=dict(color='#9370DB', width=3),
                marker=dict(size=8),
                text=compulsion_avg_scores.values,  # Show average scores on the points
                textposition='top center'
            )
        )

        # Update layout to include a second y-axis
        fig.update_layout(
            title='Number of Instances and Average Total Score by Compulsion Type',
            xaxis_title='Compulsion Type',
            yaxis_title='Number of Instances',
            yaxis2=dict(
                title='Average Total Score',
                overlaying='y',  # Overlay y-axis 2 on the same plot as y-axis 1
                side='right'
            ),
            legend=dict(x=0.1, y=1.1),
            xaxis_tickangle=45,
            yaxis=dict(showgrid=True)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
    
   ##############################################################

# Placeholder for Diagnostic Analytics
elif sidebar_option == "Diagnostic Analytics":
    st.markdown("<h1 style='color: turquoise;'>Diagnostic Analytics</h1>", unsafe_allow_html=True)
    st.write("Diagnostic Analytics will help the user to identify relationships between the various features used in this dataset related to OCD symtom severity.")

# Load dataset
    df = pd.read_csv('filtered_df.csv')

    st.markdown("""
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 5px; border: 2px solid #D3D3D3;">
            <h2 style="color: turquoise;">What are Correlations?</h2>
    """, unsafe_allow_html=True)
    st.write("""
    Correlations measure the strength and direction of the relationship between two variables. 
    The correlation coefficient ranges from -1 to 1:
    - **+1** indicates a perfect positive correlation.
    - **-1** indicates a perfect negative correlation.
    - **0** indicates no linear correlation.
    """)

# Dropdowns for selecting the metrics
    st.write("### Choose Metrics for Correlation")
    metric_a = st.selectbox("Select Metric A:", ['Age', 'Duration of Symptoms (months)', 'Total_Score'])
    metric_b = st.selectbox("Select Metric B:", ['Age', 'Duration of Symptoms (months)', 'Total_Score'])

# Calculate correlation
    if metric_a != metric_b:
        correlation = df[metric_a].corr(df[metric_b])
    
    # Explanation of correlation coefficient
        st.write(f"The correlation coefficient between **{metric_a}** and **{metric_b}** is: **{correlation:.2f}**.")
        if correlation > 0.7:
            st.write("This indicates a strong positive correlation.")
        elif 0.3 < correlation <= 0.7:
            st.write("This indicates a moderate positive correlation.")
        elif -0.3 <= correlation <= 0.3:
            st.write("This indicates a weak or no correlation.")
        elif -0.7 <= correlation < -0.3:
            st.write("This indicates a moderate negative correlation.")
        else:
            st.write("This indicates a strong negative correlation.")
    
    # Visualization using Plotly
        fig = px.scatter(df, x=metric_a, y=metric_b, trendline="ols", title=f"Correlation between {metric_a} and {metric_b}")
        fig.update_traces(line=dict(color='turquoise'))
        fig.update_layout(xaxis_title=metric_a, yaxis_title=metric_b)
        st.plotly_chart(fig)
    else:
        st.warning("Please choose different metrics for correlation.")

    st.markdown("""
        <div style="background-color: #F9F9F9; padding: 15px; border-radius: 5px; border: 2px solid #D3D3D3;">
            <h2 style="color: turquoise;">What are Associations?</h2>
    """, unsafe_allow_html=True)
    st.write("""
    Associations between variables refer to the statistical relationships or dependencies that exist between two or more variables. They help identify patterns, trends, and potential causative factors within datasets
    """)

    st.write("""
    The Chi-square test is a statistical method used to determine whether there is a significant association between two categorical variables. It assesses how the observed frequencies in a contingency table compare to the frequencies expected under the assumption of independence.
    """)

    st.write("""
    The p-value is the probability of obtaining a Chi-square statistic as extreme as, or more extreme than, the observed statistic, assuming that the null hypothesis is true.
    """)

    # Convert Total_Score to Score_Category
    def assign_score_category(df):
            df['Score_Category'] = np.where(df['Total_Score'] <= 20, 'Low', 'High')
            return df

    df = assign_score_category(df)

#Select variable for heatmap
    diagnosis_options = ['Family History of OCD', 'Anxiety Diagnosis', 'Depression Diagnosis', 'Gender']
    selected_diagnosis = st.selectbox("Select Metric:", diagnosis_options)

# Create a contingency table
    contingency_table = pd.crosstab(df['Score_Category'], df[selected_diagnosis])

# Calculate Chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

# Display Chi-square results
    st.subheader("Chi-square Test Results")
    st.write(f"Chi-square Statistic: {chi2_stat:.2f}")
    st.write(f"P-value: {p_val:.4f}")

# Generate a heatmap with turquoise scale
    plt.figure(figsize=(8, 5))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='BuGn', cbar=True)
    plt.title(f'Heatmap of Score_Category vs {selected_diagnosis}')
    plt.xlabel(selected_diagnosis)
    plt.ylabel('Score_Category')

# Display the heatmap in Streamlit
    st.pyplot(plt)

# Explanation of Chi-square test results
    if p_val < 0.05:
        st.write("The p-value is less than 0.05, indicating a statistically significant relationship between Score_Category and", selected_diagnosis)
        st.write("This suggests that the distribution of Score_Category is dependent on the selected diagnosis.")
    else:
        st.write("The p-value is greater than 0.05, indicating no statistically significant relationship between Score_Category and", selected_diagnosis)
        st.write("This suggests that the distribution of Score_Category is independent of the selected diagnosis.")
