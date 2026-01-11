import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EmpathyBot", page_icon="💙")
st.title("💙 EmpathyBot: Your Supportive Friend")

# --- 2. LOAD BLENDERBOT (Better at Empathy) ---
@st.cache_resource
def load_model():
    # This model is specifically designed for "human-like" empathetic chat
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- 3. CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLE INPUT ---
if user_input := st.chat_input("How are you feeling?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # BlenderBot works differently: We give it the inputs, and it generates a clean reply.
            inputs = tokenizer(user_input, return_tensors="pt")
            
            # Generate the reply
            reply_ids = model.generate(**inputs)
            
            # Decode the reply
            response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            
            st.markdown(response)
    
    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})