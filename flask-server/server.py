
from datetime import datetime
import os
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning import APIClient
# IBM Watson Granite Model Configuration


import os
import pandas as pd

# Function to save campaign data to Excel
def save_to_excel(data, filename="campaign_brief.xlsx"):
    """
    Save the campaign data to an Excel file.

    Args:
        data (dict): Dictionary containing campaign data.
        filename (str): Name of the Excel file.
    """
    df = pd.DataFrame([data])  # Convert dictionary to DataFrame

    if os.path.exists(filename):
        # Append to existing Excel file
        existing_df = pd.read_excel(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(filename, index=False)
    else:
        # Create a new Excel file
        df.to_excel(filename, index=False)



def initialize_watson_granite():
    """
    Initialize Watson Granite model.
    """
    # IBM Watson Machine Learning credentials and project information
    credentials = {
        "url": "https://jp-tok.ml.cloud.ibm.com",  # Replace with your WML service URL
        "apikey": "ddRZNR2OMQMSufEb81AnA87Hvn-q__ZkNJEXU3Rv5NhD"  # Replace with your WML API key
    }
    project_id = "7d624502-0abc-4210-9776-b29b0aecd77d"  # Replace with your project ID

    # Initialize model parameters
    model_id_1 = "mistralai/mixtral-8x7b-instruct-v01"#"ibm/granite-13b-instruct-v2"
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }

    # Initialize model
    try:
        granite_model = Model(model_id=model_id_1, params=parameters, credentials=credentials, project_id=project_id)
        print("Model initialized successfully.")
        return granite_model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        exit()


def load_historical_data():
    """Load historical campaign data."""
    campaign_data = pd.read_excel("/content/Sample_Data.xlsx")  # Update the path
    return campaign_data


def analyze_historical_data(campaign_data):
    """Analyze historical data for patterns and insights."""
    offer_summary = campaign_data.groupby("CAMPAIGN NAME")["Offer_Name"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    ).reset_index()
    campaign_data['Accepted'] = pd.to_numeric(campaign_data['Accepted'], errors='coerce').fillna(0)
    campaign_data['Targeted'] = pd.to_numeric(campaign_data['Targeted'], errors='coerce').fillna(0)

    acceptance_summary = campaign_data.groupby("CAMPAIGN NAME").apply(
        lambda x: (x["Accepted"].sum() / x["Targeted"].sum() * 100) if x["Targeted"].sum() > 0 else 0
    ).reset_index(name="Acceptance Rate (%)")

    insights = pd.merge(offer_summary, acceptance_summary, on="CAMPAIGN NAME")
    insights = pd.merge(insights, campaign_data[["CAMPAIGN NAME", "Data Criteria", "Reward type", "CG volume", "Frequency"]].drop_duplicates(), on="CAMPAIGN NAME")

    return insights

def generate_question(granite_model, context):
    """Generate the next question dynamically using Watson Granite."""
    try:
        prompt = (
            f"You are a chatbot interacting with a user to design a marketing campaign. "
            f"The conversation so far is:\n{context}\n"
            f"Based on this, what should be the next question? Keep it concise."
        )
        response = granite_model.generate(prompt)
        question = response["results"][0]["generated_text"].strip()
        return question
    except Exception as e:
        print(f"Failed to generate question: {e}")
        return "Sorry, I couldn't generate a question. Can you clarify your requirement?"

def generate_campaign_brief(granite_model, prompt):
    """Generate a campaign brief."""
    try:
        response = granite_model.generate(prompt)
        return response["results"][0]["generated_text"].strip()
    except Exception as e:
        print(f"Failed to generate text: {e}")
        return None
    

def chatbot_loop(granite_model, insights):
    """Interactive chatbot loop."""
    context = ""
    user_responses = []

    print("Chatbot: Hello! Let's design a marketing campaign together.")

    # Start dynamic interaction
    while True:
        # Generate the next question
        question = generate_question(granite_model, context)
        print(f"Chatbot: {question}")

        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Thanks for chatting! Goodbye.")
            break

        # Update context with user response
        context += f"Chatbot: {question}\nYou: {user_input}\n"
        user_responses.append(user_input)

        # If sufficient input, generate campaign brief
        if len(user_responses) >= 5:  # Assume we need at least 3 responses
            requirement = user_responses[-1]  # Last user input is the requirement
            campaign_prompt = prepare_campaign_prompt(requirement, insights)
            campaign_brief = generate_campaign_brief(granite_model, campaign_prompt)

            print("\nChatbot: Here's the campaign brief I generated for you:")
            print(campaign_brief)

            # Save the campaign brief and responses to Excel
            campaign_data = {
                "User Responses": user_responses,
                "Campaign Brief": campaign_brief
            }
            save_to_excel(campaign_data)

            break


def prepare_campaign_prompt(requirement, insights):
    """Create a structured prompt for campaign brief generation."""
    historical_context = "\nHistorical Insights:\n"
    for _, row in insights.iterrows():
        historical_context += (
            f"- CAMPAIGN NAME: {row['CAMPAIGN NAME']}, Most Frequent Offer: {row['Offer_Name']}, "
            f"Acceptance Rate: {row['Acceptance Rate (%)']:.2f}%, Reward Type: {row['Reward type']}, "
            f"CG Volume: {row['CG volume']}, Frequency: {row['Frequency']}\n"
        )

    prompt = (
        f"""You are tasked with creating a marketing campaign brief:\n
    Requirement: {requirement}\n
    {historical_context}\n
    Create a JSON response including:
    - "CAMPAIGN NAME"
    - "Data Criteria"
    - "Offer_Name"
    - "Reward type"
    - "CG volume"
    - "Frequency"
    - "Offer Text" (Provide a sample offer text)
    """
    )
    return prompt

def main():
    granite_model = initialize_watson_granite()  # Ensure this function initializes the model
    campaign_data = load_historical_data()
    insights = analyze_historical_data(campaign_data)
    chatbot_loop(granite_model, insights)

if __name__ == "__main__":
    main()

