import gradio as gr
from utils.image_utils import detect_image
from utils.video_utils import detect_video

# ==========================
# Premium CSS
# ==========================
css = """
.gradio-container{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

/* Hide default footer */
footer{
    visibility:hidden;
}

/* All markdown text */
.markdown,
.markdown *,
.prose,
.prose *{
    color:white !important;
}

/* Labels */
label{
    color:white !important;
}

/* Buttons */
button{
    border-radius:12px !important;
    font-weight:bold !important;
}

/* Cards */
.gr-box{
    border-radius:18px !important;
}

/* Tabs */
button[role="tab"]{
    color:white !important;
    font-weight:bold !important;
}

button[role="tab"][aria-selected="true"]{
    background:#2563eb !important;
    color:white !important;
    border-radius:10px;
}
"""

# ==========================
# APP
# ==========================

with gr.Blocks(
    title="AI Helmet Violation Monitoring System",
    theme=gr.themes.Soft(),
    css=css
) as app:

    # ==========================
    # HERO
    # ==========================

    gr.HTML("""

<div style="background:linear-gradient(90deg,#2563eb,#06b6d4);
padding:35px;
border-radius:20px;
text-align:center;
margin-bottom:20px;">

<h1 style="color:white;font-size:42px;margin:0;">
🚨 AI Helmet Violation Monitoring System
</h1>

<h3 style="color:white;">
Real-Time Helmet Detection using YOLOv11
</h3>

<p style="color:white;font-size:18px;">
AI Powered • Computer Vision • Smart Traffic Monitoring
</p>

</div>

""")

    # ==========================
    # Banner
    # ==========================

    gr.Image(
        value="assets/banner.png",
        show_label=False,
        interactive=False
    )

    # ==========================
    # Feature Cards
    # ==========================

    gr.HTML("""

<div style="display:flex;
justify-content:center;
gap:20px;
margin-top:20px;
margin-bottom:25px;">

<div style="background:#1e293b;
padding:20px;
border-radius:15px;
width:220px;
text-align:center;
box-shadow:0 0 20px rgba(0,255,255,.15);">

<h2>📷</h2>
<h3>Image Detection</h3>
<p>Upload images and detect helmets instantly.</p>

</div>

<div style="background:#1e293b;
padding:20px;
border-radius:15px;
width:220px;
text-align:center;
box-shadow:0 0 20px rgba(0,255,255,.15);">

<h2>🎥</h2>
<h3>Video Detection</h3>
<p>Process complete videos using YOLO.</p>

</div>

<div style="background:#1e293b;
padding:20px;
border-radius:15px;
width:220px;
text-align:center;
box-shadow:0 0 20px rgba(0,255,255,.15);">

<h2>🚨</h2>
<h3>Violation Monitoring</h3>
<p>Smart AI based helmet monitoring.</p>

</div>

</div>

""")

    # ==========================
    # Tabs
    # ==========================

    with gr.Tabs():
          

        # ----------------------
        # IMAGE
        # ----------------------

        with gr.Tab("📷 Image Detection"):

            gr.Markdown("## Upload an Image")

            image_input = gr.Image(
                type="pil",
                label="Input Image"
            )

            image_output = gr.Image(
                label="Detection Result"
            )

            image_btn = gr.Button(
                "🛡 Analyze Image",
                variant="primary"
            )

            image_btn.click(
                fn=detect_image,
                inputs=image_input,
                outputs=image_output
            )

        # ----------------------
        # VIDEO
        # ----------------------

        with gr.Tab("🎥 Video Detection"):

            gr.Markdown("## Upload a Video")

            video_input = gr.Video(
                label="Input Video"
            )

            video_output = gr.File(
                label="⬇ Download Processed Video"
            )

            video_btn = gr.Button(
                "🎥 Analyze Video",
                variant="primary"
            )

            video_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=video_output
            )

        # ----------------------
        # Webcam
        # ----------------------

        with gr.Tab("📹 Live Webcam"):

         gr.HTML("""

<h1 style="color:white;">📹 Live Webcam</h1>

<p style="color:white;font-size:18px;">
🚧 This feature is available in the Desktop Version.
</p>

<ul style="color:white;font-size:18px;">
<li>✅ Real-Time Webcam Detection</li>
<li>✅ Auto Evidence Capture</li>
<li>✅ CSV Logging</li>
<li>✅ Live Dashboard</li>
</ul>

""")

        # ----------------------
        # About
        # ----------------------

        with gr.Tab("ℹ️ About"):

            gr.HTML("""

<h1 style="color:white;">🚨 AI Helmet Violation Monitoring System</h1>

<h2 style="color:#38bdf8;">🛠 Tech Stack</h2>

<ul style="color:white;font-size:18px;">
<li>YOLOv11</li>
<li>OpenCV</li>
<li>Python</li>
<li>Gradio</li>
</ul>

<h2 style="color:#38bdf8;">✨ Features</h2>

<ul style="color:white;font-size:18px;">
<li>✅ Image Detection</li>
<li>✅ Video Detection</li>
<li>✅ Desktop Webcam Detection</li>
<li>✅ Auto Evidence Capture</li>
<li>✅ CSV Logging</li>
<li>✅ Live Dashboard</li>
</ul>

<hr>

<h2 style="color:#38bdf8;">👨‍💻 Developed By</h2>

<h3 style="color:white;">Ashutosh Pandey</h3>

<p style="color:white;">
AI & ML Engineer<br>
Computer Vision Enthusiast
</p>

""")

    # ==========================
    # Footer
    # ==========================

gr.HTML("""

<hr style="border:1px solid #334155; margin-top:30px;">

<div style="
text-align:center;
padding:30px;
">

<h2 style="color:#38bdf8; margin-bottom:10px;">
👨‍💻 Developed by Ashutosh Pandey
</h2>

<p style="color:white; font-size:18px;">
AI & Machine Learning • Computer Vision • Deep Learning
</p>

<p style="color:#94a3b8;">
Built with ❤️ using Python, YOLOv11, OpenCV & Gradio
</p>

<p style="color:#38bdf8; font-weight:bold;">
🚀 Hugging Face Deployment Ready
</p>

</div>

""")

app.launch()