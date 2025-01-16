import os
import signal
import csv
import matplotlib.pyplot as plt
import streamlit as st

def terminate_process_group(process):
    """
    Terminates a process group of a given process.
    Cross-platform handling for Windows and Unix-like systems.
    """
    try:
        if os.name == 'nt':  # Windows
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:  # Unix-like systems
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception as e:
        st.error(f"Error terminating process group: {e}")


def display_training_progress(results_file, total_epochs, training_process):
    """
    Displays a dynamic graph, table, and terminal-style logs during training.
    Handles encoding issues by replacing undecodable characters.
    """
    import time

    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    log_placeholder = st.empty()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metrics")
    ax.set_title("Training Progress")
    ax.grid(True)

    epochs, mAP50, mAP50_95 = [], [], []
    log_lines = []

    line1, = ax.plot([], [], '-o', color='lightblue', label="mAP50")
    line2, = ax.plot([], [], '-o', color='lightcoral', label="mAP50-95")
    ax.legend()

    chart_placeholder.pyplot(fig)

    last_modified_time = 0
    while training_process.poll() is None:
        try:
            # Display Training Output log in the form of a terminal:
            output_line = training_process.stdout.readline()
            if output_line:
                log_lines.append(output_line.strip())
                if len(log_lines) > 5:
                    log_lines.pop(0)
                log_placeholder.markdown(
                    f"<div style='background-color: black; color: white; font-family: monospace; padding: 10px; height: 150px; overflow-y: auto;'>{'<br>'.join(log_lines)}</div>",
                    unsafe_allow_html=True,
                )
        except UnicodeDecodeError as e:
            log_placeholder.markdown(
                f"<div style='background-color: black; color: red; font-family: monospace; padding: 10px;'>Decoding Error: {e}</div>",
                unsafe_allow_html=True,
            )

        # Checks training results and updates based on each evaluated epoch. 
        if os.path.exists(results_file):
            modified_time = os.path.getmtime(results_file)
            if modified_time > last_modified_time:
                last_modified_time = modified_time
                try:
                    with open(results_file, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        epochs, mAP50, mAP50_95 = [], [], []
                        for row in reader:
                            epochs.append(int(row["epoch"]))
                            mAP50.append(float(row["metrics/mAP50(B)"]))
                            mAP50_95.append(float(row["metrics/mAP50-95(B)"]))

                    # Graph Data Update
                    line1.set_data(epochs, mAP50)
                    line2.set_data(epochs, mAP50_95)
                    ax.relim()
                    ax.autoscale_view()
                    chart_placeholder.pyplot(fig)

                    # Table data Update
                    if epochs:
                        table_placeholder.table({
                            "Metric": ["Epoch", "mAP50", "mAP50-95"],
                            "Value": [epochs[-1], mAP50[-1], mAP50_95[-1]],
                        })
                except Exception as e:
                    st.error(f"Error reading results file: {e}")

        time.sleep(0.1)

    st.success("Training completed!")
