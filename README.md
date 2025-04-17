# HealthChatAssistant
A chat assistant made up of langchain agents that answer health-related questions using a Medicare/Medicare dataset pertaining to location, measures for conditions, scores, etc.

The SQL AGENT WILL TAKE 4-6 maybe even up to 10 minutes to generate and complete the query depending on the question. This is because schema/column_descriptions/extra sql tools could be further optimized.

Download both files into a new folder/directory. 
cd into that folder and run streamlit run main.py. Make sure you have python and streamlit installed. 

Dependencies/libraries utilized should be attached in a separate requirements.txt. Download requirements.txt file and in the project directory run pip install -r /path/to/requirements.txt to have all necessary packages installed.

Please note that clicking reset chat does not actually reset the chat, it resets the chat log.conversation memory, but reloading the website will keep the previous prompts and text.

When downloading chat log, you must enter more than two prompts for the json file to be populated.


