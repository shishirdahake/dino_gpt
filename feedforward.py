from dinohelper import Decoder, generate_name
import torch
import streamlit as st

# Load Model weights from dino_gpt
model_weights = torch.load('dino_gpt.pth')

# Load the model
model = Decoder()
# Add model weights from state dict and put model in eval (no learning) mode
model.load_state_dict(model_weights)
model.eval()

st.title("DinoGPT")
seed = st.text_input("Lets name our Dinosaur with DinoGPT")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True



st.button('Generate Name', icon='🦖', on_click=click_button)

if st.session_state.clicked:
    dino_name = generate_name(model, seed)
    st.success(f'Our latest Dinosaur is named **{dino_name.upper()}**. Roar!')

st.divider()
st.caption(
    "© 2026 Shishir Dahake · Apache-2.0 · [Source on GitHub](https://github.com/shishirdahake/dino_gpt)"
)