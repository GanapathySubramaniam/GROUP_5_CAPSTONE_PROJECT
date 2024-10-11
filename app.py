import streamlit as st
from video_combiner import combine_mp4_videos
from animator import process_video

def generate_video(input_text):
    texts=input_text.split()
    texts=[text.lower() for text in texts]
    video_files = [f'real_vids/{text}.mp4' for text in texts]
    print(video_files)
    for i in video_files:
        process_video(i,output_folder)
    animated_vids=[f'hpe_vids/animated_{text}.mp4' for text in texts]
    print(animated_vids)
    combine_mp4_videos(animated_vids,input_text)


st.title("Sign Language Generator")

output_folder = 'hpe_vids'

input_text = st.text_input("Enter a text:")

if st.button("Generate"):
    with st.spinner("Generating video..."):
        generate_video(input_text)
    st.balloons()
    st.video("combined_output.mp4")



