# Imports
import streamlit as st
from streamlit_option_menu import option_menu

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
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Our prediction model is based on historical claims data and uses advanced machine learning techniques to generate the results. While the predictions
            are designed to assist the WCB, they are not definitive decisions and should be used alongside human judgment in the claim review process.<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let’s begin by entering the details of a new claim, and let’s see how <b>GrantApp</b> can help you make predictions more efficiently!<br>
            </p>
            """, unsafe_allow_html=True 
        )


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
