import streamlit as st
import folium
from streamlit_folium import st_folium

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = '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' # Remember to set this environment variable

    # Set up columns for chatbot and map
    chat_col, map_col = st.columns([1, 2])  # Adjust ratio as desired
    
    with chat_col:

        # Title and greeting message for the chatbot
        st.title("Chat with Campus Assistant!")
        st.write("Hello! I'm your friendly Door2Door chatbot. I can help answer your questions, provide information, or just chat abour our beautiful campus. Let's start our conversation!")

        model = 'llama3-8b-8192'
        memory = ConversationBufferWindowMemory(k=7, memory_key="chat_history", return_messages=True)

        user_question = st.text_input("Ask a question:")

        # Session state variable to keep track of chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        else:
            for message in st.session_state.chat_history:
                memory.save_context(
                    {'input': message['human']},
                    {'output': message['AI']}
                )

        # Initialize Groq Langchain chat object and conversation
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
        )

        # If the user has asked a question,
        if user_question:
            # Construct a chat prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content="You are a Yonsei Campus assistant"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Create a conversation chain
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )
            
            # Generate chatbot response
            response = conversation.predict(human_input=user_question)
            message = {'human': user_question, 'AI': response}
            st.session_state.chat_history.append(message)
            st.write("Chatbot:", response)

    with map_col:
        st.header("Map of Yonsei University")
        
        # Example coordinates for Yonsei University
        yonsei_lat, yonsei_lon = 37.5651, 126.9386
        folium_map = folium.Map(location=[yonsei_lat, yonsei_lon], zoom_start=16)
        folium.Marker([yonsei_lat, yonsei_lon], popup="Yonsei University").add_to(folium_map)
        
        # Render the map
        st_folium(folium_map, width=700, height=500)

if __name__ == "__main__":
    main()
