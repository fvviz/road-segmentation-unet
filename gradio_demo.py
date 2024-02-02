import gradio as gr
from sliding_window import run_sliding_window_pil


iface = gr.Interface(
    fn=run_sliding_window_pil,
    inputs=[gr.Image(type="pil", label="Input Image"), 
            gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Threshold"),
             gr.Dropdown(
            [256, 128, 64, 32], label="Sliding window size", value=256
        ),
        gr.Dropdown(
            [256, 128, 64, 32], label="Stride size", value=256
        )]
            ,
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Satellite road detection"
)

iface.launch()
