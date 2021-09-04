import streamlit_drawable_canvas as st_canvas
import streamlit as st
import os
import app_utils as utils
from PIL import Image


st.set_page_config("Convolutional VAE")
st.title("Convolutional VAE")

st.markdown("Simple streamlit app for Convolutional VAEs")


st.header("Image Reconstruction", "recon")

with st.form("reconstruction"):
    model_name = "mnist_model.ckpt"
    recon_canvas = st_canvas.st_canvas(
        # Fixed fill color with some opacity
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=8,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="recon_canvas",
    )
    submit = st.form_submit_button("Perform Reconstruction")
    if submit:
        reoncn_model = utils.load_model(model_name)
        inp_tens = utils.canvas_to_tensor(recon_canvas)
        img, _, _ = reoncn_model(inp_tens)
        img = (img+1)/2
        out_img = utils.resize_img(utils.tensor_to_img(img), 150, 150)
if submit:
    st.image(out_img)

st.header("Image interpolation", "interpolate")
with st.form("interpolation"):
    model_name = "mnist_model.ckpt"
    stroke_width = 8
    cols = st.columns([1, 3, 2, 3, 1])
    with cols[1]:
        # create the canvas
        canvas_result_1 = st_canvas.st_canvas(
            fill_color = "rgba(255, 165, 0, 0.3)",
            stroke_width = stroke_width,
            stroke_color = "#FFFFFF",
            background_color = "#000000",
            update_streamlit = True,
            height = 150,
            width = 150,
            drawing_mode = "freedraw",
            key = "canvas1",
            )
    with cols[3]:# create the canvas
        canvas_result_2 = st_canvas.st_canvas(
            fill_color = "rgba(255, 165, 0, 0.3)",
            stroke_width = stroke_width,
            stroke_color = "#FFFFFF",
            background_color = "#000000",
            update_streamlit = True,
            height = 150,
            width = 150,
            drawing_mode = "freedraw",
            key = "canvas2",
            )
        submit = st.form_submit_button("Perform Interpolation")
        if submit:
            model = utils.load_model(model_name)
            tens1 = utils.canvas_to_tensor(canvas_result_1)
            tens2 = utils.canvas_to_tensor(canvas_result_2)
            interplated_output = utils.interpolation(
                model, tens1, tens2
                )
if submit:
    st.image(interplated_output)


