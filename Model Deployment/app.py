import import_ipynb
import pandas as pd
import streamlit as st
# Importing the necessary methods from the Jupyter notebook
from model import DQNAgent, QLearningAgent  # Import from model.ipynb


# Streamlit interface for cryptocurrency price prediction
def main():
    st.title("Cryptocurrency Price Prediction with Q-Learning / DQN")
    st.write("Use Q-learning or Deep Q-learning to predict buy/hold/sell actions based on cryptocurrency data.")

    # User input for date and time (example format: 2024-11-29 19:15:00)
    user_input = st.text_input("Enter Date and Time (yyyy-mm-dd hh:mm:ss):")

    if user_input:
        # Load your data (example data loading, replace with your actual data)
        df = pd.read_csv("Shiba.csv")  # Ensure you have your dataset in this file path

        # Extract the state for the given date and time
        state = df[df["Open_time"] == user_input][["Open", "High", "Low", "Close", "Volume"]].values[1]

        # Example state-space and action-space configuration
        action_space = 3  # Buy, hold, sell
        state_space = len(state)  # Number of features in the state

        # Choose the agent type (Q-learning or DQN)
        agent_choice = st.radio("Choose Agent Type:", ("Q-learning", "Deep Q-learning"))

        if agent_choice == "Q-learning":
            agent = QLearningAgent(action_space, state_space, bins=5)

        elif agent_choice == "Deep Q-learning":
            agent = DQNAgent(state_space, action_space)

        # Make prediction based on the agent's learned behavior
        action = agent.choose_action(state)
        action_text = ["Buy", "Hold", "Sell"][action]  # Convert to action text

        # Display the prediction
        st.write(f"Predicted Action: {action_text}")

if __name__ == "__main__":
    main()
