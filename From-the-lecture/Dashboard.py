import streamlit as st

st.set_page_config(
    page_title="OCD SEVERITY PREDICTOR"
 
)

# Sidebar configuration
st.sidebar.image("./From-the-lecture/assets/OCD.jpeg",)
st.sidebar.success("Select a tab above.")

# Page information
st.markdown(
    "<h1 style='color: turquoise;'>OCD PREDICTOR</h1>",
    unsafe_allow_html=True
)


# Input Form for user details
st.markdown("<h2 style='color: turquoise;'>Patient Form</h2>", unsafe_allow_html=True)


with st.form(key='patient_info_form'):
    # New input fields
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])  # Gender selection
    family_history = st.selectbox("Family History of OCD?", options=["Yes", "No"])  # Family history
    depression = st.selectbox("Does Patient have depression?", options=["Yes", "No"])  # Depression status
    anxiety = st.selectbox("Does Patient have anxiety?", options=["Yes", "No"])  # Anxiety status
    age = st.number_input("Age", min_value=0, max_value=120, value=25)  # Default age
    

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.success("Information Submitted Successfully!")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Family History of OCD:** {family_history}")
    st.write(f"**Depression:** {depression}")
    st.write(f"**Anxiety:** {anxiety}")
    st.write(f"**Age:** {age}")
  




# You can also add text right into the web as long comments (""")


### UNCOMMENT THE CODE BELOW TO SEE EXAMPLE OF INPUT WIDGETS

# # DATAFRAME MANAGEMENT
# import numpy as np

# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)

# # Add a slider to the sidebar:
# add_slider = st.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )




predicted_severity = "Severe Symptoms"  # Directly stating the predicted severity
st.markdown(f"### Predicted: **{predicted_severity}**")

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

#  test


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# Create an enhanced illustration of high symptoms
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


