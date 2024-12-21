# Imports
import streamlit as st
from streamlit_option_menu import option_menu
from data_pipeline import *

import pandas as pd

# Prediction function
def make_predictions(data_input):
    for n_fold in range(6):
        df = pipeline(data_input, n_fold)
        all_folds_predictions = predict_probability(df, n_fold)
        
        # Define targets mapping
    class_mapping = {
        0:'1. CANCELLED', 
        1:'2. NON-COMP',
        2:'3. MED ONLY', 
        3:'4. TEMPORARY',
        4:'5. PPD SCH LOSS', 
        5:'6. PPD NSL', 
        6:'7. PTD', 
        7:'8. DEATH'
    }

    final_test_preds = np.argmax(all_folds_predictions / 6, axis=1)
    predictions_df = pd.DataFrame({
        'Claim Identifier': df.index,
        'Claim Injury Type': final_test_preds
    })
    predictions_df["Claim Injury Type"] = predictions_df["Claim Injury Type"].replace(class_mapping)

    st.dataframe(predictions_df)




def get_session_state():
    """
    Utility function to retrieve the Streamlit session state.
    """
    return st.session_state

class GrantApp:
    def __init__(self):
        """
        Initialize the GrantApp class.
        """
        self.session_state = get_session_state()

    def display_home_page(self):
        """
        Display the Home page content.
        """
        st.title("Welcome to the GrantApp")

        st.markdown(
            """
            <p style="color: #465e54; font-size: 20px;">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Welcome to the <b>GrantApp</b>, an innovative tool designed to help automate the decision-making process
            for workers' compensation claims. This app leverages the power of machine learning to assist the New York Workers' Compensation Board (WCB)
            in managing and reviewing claims more efficiently. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The WCB has reviewed millions of claims over the years, but manually processing each one is a time-consuming and
            complex task. <b>GrantApp</b>* aims to automate this process by providing a <b>prediction model</b> that can help streamline the decision-making
            for new claims based on historical data.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; On this platform, you can navigate to different sections: <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b> - Problem</b>: Understand the problem we are addressing.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b> - Prediction</b>: Get predictions for workers' compensation claims based on the model.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let's dive into the app and explore how we can help make the claims process faster and more efficient!
            </p>
            """, unsafe_allow_html=True 
        )

        
    def display_problem(self):
        """
        Display information about the problem we are trying to solve.
        """
        st.title("The Problem We Are Solving")

        st.markdown(
            """
            <p style="color: #465e54; font-size: 20px;">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The <b>New York Workers’ Compensation Board (WCB)</b> is responsible for overseeing the administration of workers' compensation benefits.
            This includes benefits for injured workers, as well as compensation for disability, volunteer firefighters, ambulance workers, and civil defense workers. <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The WCB has reviewed more than <b>5 million claims</b> since the year 2000, which are submitted by workers who have experienced workplace injuries.
            Despite the significant effort invested in processing these claims, the manual review process is time-consuming and inefficient, leading to delays in claim decisions.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The need for automation arises to improve the speed and accuracy of decision-making. By automating the claim review process, we can help the WCB
            make faster and more reliable decisions, ensuring injured workers receive timely assistance.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is where <b>GrantApp</b> comes in! Our goal is to build a model that can accurately predict the outcomes of workers' compensation claims,
            streamlining the decision-making process and supporting the WCB's mission of delivering better service to injured workers.<br>
            </p>
            """, unsafe_allow_html=True 
        )


    def display_prediction(self):
        """
        Use user input to create a prediction.
        """
        st.title("Predict Workers' Compensation Claim Outcome")

        st.markdown(
            """
            <p style="color: #465e54; font-size: 20px;">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Welcome to the <b>Prediction</b> page! Here, you can input details about a workers' compensation claim to receive a prediction about
            its outcome. The model we use has been trained on historical data and is designed to assist the New York Workers' Compensation Board in making more efficient and accurate decisions.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To get started, please provide the necessary details about the claim. Based on the input you provide, the model will predict the class in which the
            claim belongs to, helping the WCB make data-driven decisions and expedite the claims process.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let’s begin by entering the details of a new claim, and let’s see how <b>GrantApp</b> can help you make predictions more efficiently!<br>
            </p>
            """, unsafe_allow_html=True
        )

        selected = option_menu(
            "Select Input Method",
            ["Input Manually", "Upload CSV"],
            menu_icon="cast", default_index=0
        )

        data_input = None  # Initialize variable to ensure it's defined

        if selected == "Upload CSV":
            st.markdown('<p style="color: #465e54; font-size: 20px;">Choose a CSV file to upload:</p>', unsafe_allow_html=True)
            uploaded_csv = st.file_uploader("", type="csv")

            if uploaded_csv is not None:
                try:
                    data_input = pd.read_csv(uploaded_csv)
                    st.success("File uploaded successfully!")

                    st.markdown('<p style="color: #465e54; font-size: 20px;">Here is a preview of your uploaded CSV:</p>', unsafe_allow_html=True)
                    st.dataframe(data_input)

                    st.markdown(f'<p style="color: #465e54; font-size: 16px;">The CSV has {data_input.shape[0]} rows and {data_input.shape[1]} columns.</p>', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<p style="color: red; font-size: 16px;">An error occurred while reading the file: {e}</p>', unsafe_allow_html=True)

        elif selected == "Input Manually":
            st.markdown('<p style="color: #465e54; font-size: 20px;">Enter data manually, one row per line.</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #465e54; font-size: 15px;">Enter data (comma separated)</p>', unsafe_allow_html=True)

            manual_input = st.text_area("", height=200)

            if manual_input:
                try:
                    rows = [line.split(',') for line in manual_input.strip().split('\n')]
                    columns = [f"Column {i+1}" for i in range(len(rows[0]))]
                    data_input = pd.DataFrame(rows, columns=columns)

                    st.success("Data entered successfully!")
                    st.markdown('<p style="color: #465e54; font-size: 20px;">Here is the manually entered data:</p>', unsafe_allow_html=True)
                    st.dataframe(data_input)

                    st.markdown(f'<p style="color: #465e54; font-size: 16px;">The entered data has {data_input.shape[0]} rows and {data_input.shape[1]} columns.</p>', unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f'<p style="color: red; font-size: 16px;">An error occurred while processing the input data: {e}</p>', unsafe_allow_html=True)

        # Create a Predict button
        if st.button("Predict"):
            if data_input is not None:
                prediction = make_predictions(data_input)  # Call prediction function
                st.success("Prediction made successfully!")
            else:
                st.error("Please provide input data before making a prediction.")


    def run(self):
            """
            Run the GrantApp application.
            """
            st.set_page_config(page_title="GrantApp", page_icon="🔎", layout="wide")

            # Display the option_menu without setting the default
            selected_option = option_menu(
                menu_title=None,
                options=["Home", "Problem", "Prediction"],
                icons=["house", "question", "search"],
                orientation="horizontal",
                styles={
                    # Container styling
                    "container": {"padding": "0!important", "color": "#ffffff"},
                },
            )

            # Store the selected option in session_state
            self.session_state.page = selected_option

            # Display content based on the selected page
            if selected_option == "Home":
                self.display_home_page()
            elif selected_option == "Problem":
                self.display_problem()
            elif selected_option == "Prediction":
                self.display_prediction()


if __name__ == "__main__":    
    app = GrantApp()
    app.run()
