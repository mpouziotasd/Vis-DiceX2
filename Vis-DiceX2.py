import streamlit as st
import cv2
import os
import time
import pandas as pd
from utils.model_utilities import load_model, detect, evaluate_model, train_model
from utils.streamlit_utilities import display_training_progress, terminate_process_group
from utils.img_processing import get_dice_analytics
from utils.img_utils import draw_fps, draw_data
import altair as alt
import subprocess

st.set_page_config(layout="wide")

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Inference"
if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "training_params" not in st.session_state:
    st.session_state.training_params = {}
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "frozen_frame" not in st.session_state:
    st.session_state.frozen_frame = None
if "is_frozen" not in st.session_state:
    st.session_state.is_frozen = False
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "model" not in st.session_state:
    st.session_state.model = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Inference"
if "inference_running" not in st.session_state:
    st.session_state.inference_running = False
if "experimental_rerun" not in st.session_state:
    st.session_state.experimental_rerun = False


def start_camera():
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access the camera")
            return False
        st.session_state.cap = cap
    return True

def release_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

def start_inference():
    if not start_camera():
        return

    st.title("Webcam Detection Streamlit App")
    freeze_toggle = st.button("Freeze/Unfreeze")
    parameter_col, display_col = st.columns([1, 3])

    with parameter_col:
        models = [file.split('.')[0] for file in os.listdir('models/') if file.endswith('.pt')]
        model_name = st.selectbox("Select Detection Model", models)
        if model_name != st.session_state.model_name:
            st.session_state.model_name = model_name
            st.session_state.model = load_model(f"models/{model_name}.pt")

    frame_placeholder = display_col.empty()

    if "prev_time" not in st.session_state:
        st.session_state.prev_time = time.time()

    if freeze_toggle:
        st.session_state.is_frozen = not st.session_state.is_frozen

    while True:
        if st.session_state.is_frozen:
            if st.session_state.frozen_frame is not None:
                frame_placeholder.image(st.session_state.frozen_frame, channels="BGR")
            else:
                st.warning("No frame to freeze yet.")
            continue

        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to read frame from the camera")
            release_camera()
            break

        results = detect(st.session_state.model, frame, device='0')[0]
        bboxes = results.boxes.xyxy.cpu().numpy()
        dice_nums = results.boxes.cls
        dice_analytics = get_dice_analytics(frame, bboxes, dice_nums)

        current_time = time.time()
        elapsed_time = current_time - st.session_state.prev_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
        st.session_state.prev_time = current_time

        draw_fps(frame, fps)
        processed_frame = draw_data(frame, bboxes, dice_nums, dice_analytics) if bboxes.size > 0 else frame
        st.session_state.frozen_frame = processed_frame

        frame_placeholder.image(processed_frame, channels="BGR")

def reset_state():
    """
    Description:
        Reset session states to default values. 
        To-do: Stop/Pause/Maintain training if mode is changed 
    """
    for key in ["cfg", "data", "epochs", "imgsz", "batch", "name", "training_process", "training_active"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.training_active = False
    st.session_state.training_params = {}
            

def training_mode():
    """
    Description:
        Main function to handle the YOLO model training UI and workflow.
    """
    st.title("YOLO Model Training")
    input_col, param_col, chart_col = st.columns([1, 1, 2])

    if "training_process" not in st.session_state:
        st.session_state.training_process = None
    if "training_params" not in st.session_state:
        st.session_state.training_params = {}
    if "training_active" not in st.session_state:
        st.session_state.training_active = False

    def reset_training():
        """
        Description:
            Stops training and clears the UI.
        """
        if st.session_state.training_process:
            terminate_process_group(st.session_state.training_process)
            st.session_state.training_process = None
        st.session_state.training_active = False
        st.session_state.training_params = {}

    if st.button("Stop Training"):
        reset_training()

    if not st.session_state.training_active:
        with input_col:
            st.subheader("Training Input")
            cfg = st.text_input("Model Configuration File", "cfg/yolov8-M.yaml", key="cfg")
            data = st.text_input("Dataset YAML File", "data/Vis-DiceX2.yaml", key="data")
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=500, value=150, step=10, key="epochs")
            imgsz = st.number_input("Image Size", min_value=64, max_value=1024, value=640, step=64, key="imgsz")
            batch = st.number_input("Batch Size", min_value=1, max_value=64, value=2, step=1, key="batch")
            name = st.text_input("Training Session Name", "MyTrain", key="name")

            count = 1
            unique_name = name
            while os.path.exists(f"runs/detect/{unique_name}"):
                count += 1
                unique_name = f"{name}{count}"
            results_file = f"runs/detect/{unique_name}/results.csv"

            command = f"python train.py --cfg {cfg} --data {data} --epochs {epochs} --imgsz {imgsz} --batch {batch} --name {unique_name}"

            if st.button("Start Training") and st.session_state.training_process is None:
                st.session_state.training_active = True
                st.session_state.training_params = {
                    "cfg": cfg,
                    "data": data,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "unique_name": unique_name,
                }

                if os.name == 'nt':  # Windows
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                    preexec_fn = None
                else:  # Unix-like systems
                    creationflags = 0
                    preexec_fn = os.setsid

                st.session_state.training_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=True,
                    encoding="utf-8", errors="replace",
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
    
    if st.session_state.training_active:
        training_params = st.session_state.training_params

        with param_col:
            st.subheader("Entered Parameters")
            st.write(f"**Configuration File:** {training_params['cfg']}")
            st.write(f"**Dataset:** {training_params['data']}")
            st.write(f"**Epochs:** {training_params['epochs']}")
            st.write(f"**Image Size:** {training_params['imgsz']}")
            st.write(f"**Batch Size:** {training_params['batch']}")
            st.write(f"**Session Name:** {training_params['unique_name']}")

        with chart_col:
            st.subheader("Training Progress")
            if st.session_state.training_process:
                display_training_progress(results_file, epochs, st.session_state.training_process)


def evaluation_mode():
    st.title("Model Evaluation")
    model_files = [file for file in os.listdir("models") if file.endswith(".pt")] # Fetches models in models/ folder
    if not model_files:
        st.warning("No model files found in the 'models' directory.")
        return
    selected_model = st.selectbox("Select Model to Evaluate", model_files)

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = pd.DataFrame(
            columns=["Model", "latency", "mAP50-95"]
        )

    start_evaluation = st.button("Evaluate Model")
    if start_evaluation:
        if selected_model:
            weights_path = os.path.join("models", selected_model)
            st.write(f"Evaluating model: `{selected_model}`")

            with st.spinner("Evaluating..."):
                try:
                    metrics = evaluate_model(weights_path)

                    if metrics["Model"] in st.session_state.evaluation_results["Model"].values:
                        st.warning(f"Model `{metrics['Model']}` has already been evaluated.")
                    else:
                        new_row = pd.DataFrame([metrics])
                        st.session_state.evaluation_results = pd.concat(
                            [st.session_state.evaluation_results, new_row], ignore_index=True
                        )
                        st.success(f"Evaluation complete for `{metrics['Model']}`")
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {e}")
        else:
            st.warning("No model selected for evaluation.")

    if not st.session_state.evaluation_results.empty:
        chart = alt.Chart(st.session_state.evaluation_results).mark_circle(size=100).encode(
            x=alt.X("latency:Q", title="Latency (ms)", scale=alt.Scale(zero=False)),
            y=alt.Y("mAP50-95:Q", title="mAP50-95", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", title="Model"),
            tooltip=["Model", "latency", "mAP50-95"],
        ).properties(
            width=600,
            height=400,
            title="Model Performance: mAP50-95 vs Latency",
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No models have been evaluated yet.")



def main():

    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("Choose an option", ["Inference", "Training", "Evaluation"])

    if mode != st.session_state.current_mode:
        reset_state()
        st.session_state.current_mode = mode

    if mode == "Inference":
        start_inference()
    elif mode == "Training":
        training_mode()
    elif mode == "Evaluation":
        evaluation_mode()
    


if __name__ == "__main__":
    main()
