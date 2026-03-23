import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import io
import base64

# Page Config
st.set_page_config(page_title="Color Tracking Tool", layout="wide")

# Title
st.title("Color Tracking Tool")
st.markdown("""
This tool allows you to track color intensity (RGB or HSV) over time across multiple images.
1. **Upload** your images.
2. **Select** each image and **Draw** a single Region of Interest (ROI).
3. **Set Times** for each image.
4. **Analyze** to generate plots and data.
""")

# Initialize Session State
if "rois" not in st.session_state:
    st.session_state.rois = {}  # {filename: [(x1, y1, x2, y2)]} (Always 1 item max)
if "drawing_states" not in st.session_state:
    st.session_state.drawing_states = {} # {filename: json_dict}
if "canvas_key_suffix" not in st.session_state:
    st.session_state.canvas_key_suffix = 0 # Used to force frontend canvas remounts

# --- Sidebar: File Upload & Settings ---
st.sidebar.header("1. Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Upload Images", 
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

st.sidebar.header("Settings")
st.sidebar.info("Upload images to begin analysis.")

# Caching mechanism using cache_resource to keep objects in memory
@st.cache_resource(show_spinner=False)
def load_images_resource(files_data):
    processed_images = {}
    for name, bdata in files_data:
        # Convert to numpy array
        img_array = np.frombuffer(bdata, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            processed_images[name] = {
                'pil': pil_img, 
                'bgr': img_bgr, 
                'rgb': img_rgb
            }
    return processed_images

if not uploaded_files:
    st.info("Please upload images to begin.")
    st.stop()
else:
    # Sort files to ensure consistent order
    uploaded_files.sort(key=lambda x: x.name)
    
    # Prepare data for cache
    files_data = []
    for f in uploaded_files:
        val = f.getvalue() 
        files_data.append((f.name, val))
        
    with st.spinner("Processing images..."):
        image_cache = load_images_resource(files_data)

    st.success(f"Loaded {len(image_cache)} images.")

# --- Step 2: ROI Selection (Per Image) ---
st.header("2. ROI Selection")

loaded_filenames = list(image_cache.keys())
n_images = len(loaded_filenames)

if n_images == 0:
    st.error("No valid images found.")
    st.stop()

# Set up slider session state tracking to allow for resets
if "selected_img_name" not in st.session_state:
    st.session_state.selected_img_name = loaded_filenames[0]

# Image Scrubber
if n_images > 1:
    selected_name = st.select_slider(
        "Select Image", 
        options=loaded_filenames, 
        key="selected_img_name",
        format_func=lambda x: f"Image {loaded_filenames.index(x)+1}: {x}"
    )
else:
    selected_name = loaded_filenames[0]

# Get Image Data
img_data = image_cache[selected_name]
pil_image = img_data['pil'] 
img_w, img_h = pil_image.size

# Scaling Logic
max_canvas_width = 800
scale_factor = 1.0

if img_w > max_canvas_width:
    scale_factor = max_canvas_width / img_w
    canvas_width = max_canvas_width
    canvas_height = int(img_h * scale_factor)
else:
    canvas_width = img_w
    canvas_height = img_h

st.caption(f"Drawing on: {selected_name} | Size: {img_w}x{img_h} | **Single ROI Enforced**")

# Tool Selection
col_tool, col_info = st.columns([1, 2])
with col_tool:
    tool_mode = st.radio("Tool Mode", ["Draw ROI", "Edit/Move ROI"], horizontal=True)

if tool_mode == "Draw ROI":
    drawing_mode = "rect"
else:
    drawing_mode = "transform"

# Callback function to handle deleting ROIs BEFORE the page renders
def clear_all_rois(first_file):
    st.session_state.rois = {}
    st.session_state.drawing_states = {}
    st.session_state.canvas_init_state = None
    st.session_state.last_selected_name = None
    st.session_state.canvas_key_suffix += 1 # Force clear the frontend
    if first_file:
        st.session_state.selected_img_name = first_file

# Copy ROI Buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    if st.button("Copy Previous"):
        curr_idx = loaded_filenames.index(selected_name)
        if curr_idx > 0:
            prev_name = loaded_filenames[curr_idx - 1]
            prev_rois = st.session_state.rois.get(prev_name, [])
            if prev_rois:
                st.session_state.rois[selected_name] = prev_rois
                # Force regeneration
                if selected_name in st.session_state.drawing_states:
                    del st.session_state.drawing_states[selected_name]
                st.session_state.last_selected_name = None 
                st.session_state.canvas_key_suffix += 1
                st.toast("Copied ROI.")
                st.rerun()
            else:
                st.warning("No ROI in previous.")
        else:
            st.warning("First image.")

with col2:
    if st.button("Apply to ALL"):
        current_rois = st.session_state.rois.get(selected_name, [])
        if current_rois:
            for name in loaded_filenames:
                st.session_state.rois[name] = current_rois
                # Clear drawing states to force refresh
                if name in st.session_state.drawing_states:
                    del st.session_state.drawing_states[name]
            
            # Reset canvas state
            st.session_state.last_selected_name = None
            st.session_state.canvas_key_suffix += 1
            st.toast(f"Applied ROI to {len(loaded_filenames)} images.")
            st.rerun()
        else:
            st.warning("No ROI to apply.")

with col3:
    # Use the callback to update the widget state cleanly
    first_file = loaded_filenames[0] if n_images > 1 else None
    if st.button("Delete All ROIs", type="primary", on_click=clear_all_rois, args=(first_file,)):
        st.toast("Cleared all ROIs.")

# Helper to generate Background Config
def get_background_config(pil_img, w, h):
    resized_pil = pil_img.resize((w, h))
    img_buffer = io.BytesIO()
    resized_pil.save(img_buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return {
        "type": "image",
        "src": f"data:image/jpeg;base64,{img_str}",
        "left": 0, "top": 0,
        "width": w, "height": h,
        "originX": "left", "originY": "top"
    }

# --- State Management for Canvas Stability ---
if "last_selected_name" not in st.session_state:
    st.session_state.last_selected_name = None
if "last_tool_mode" not in st.session_state:
    st.session_state.last_tool_mode = None
if "canvas_init_state" not in st.session_state:
    st.session_state.canvas_init_state = None

# Check triggers
inputs_changed = (selected_name != st.session_state.last_selected_name) or \
                 (tool_mode != st.session_state.last_tool_mode)

if inputs_changed or st.session_state.canvas_init_state is None:
    # 1. Retrieve or Generate State
    current_state = st.session_state.drawing_states.get(selected_name)
    
    if current_state is None:
        # Generate Fresh
        objects = []
        saved_rois = st.session_state.rois.get(selected_name, [])
        for (x1, y1, x2, y2) in saved_rois: # Will only be max 1 item
             objects.append({
                "type": "rect",
                "left": x1 * scale_factor,
                "top": y1 * scale_factor,
                "width": (x2 - x1) * scale_factor,
                "height": (y2 - y1) * scale_factor,
                "fill": "rgba(0, 255, 0, 0.1)",
                "stroke": "#00FF00",
                "strokeWidth": 2,
                "scaleX": 1, "scaleY": 1,
                "angle": 0, "opacity": 1
            })
        
        bg_config = get_background_config(pil_image, canvas_width, canvas_height)
        current_state = {
            "version": "4.4.0",
            "objects": objects,
            "backgroundImage": bg_config
        }
        st.session_state.drawing_states[selected_name] = current_state
    else:
        # Ensure Background is present
        if not current_state.get("backgroundImage"):
            current_state["backgroundImage"] = get_background_config(pil_image, canvas_width, canvas_height)

    # 2. Update Frozen State
    st.session_state.canvas_init_state = current_state
    
    # 3. Update Checkpoints
    st.session_state.last_selected_name = selected_name
    st.session_state.last_tool_mode = tool_mode

# --- Render Canvas ---
# Incorporate the suffix into the key so we can force a remount when necessary
canvas_key = f"canvas_{selected_name}_{tool_mode}_{st.session_state.canvas_key_suffix}"

canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.1)",
    stroke_width=2,
    stroke_color="#00FF00",
    background_image=None, 
    initial_drawing=st.session_state.canvas_init_state, 
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=drawing_mode,
    display_toolbar=True,
    key=canvas_key,
)

# --- Handle Updates ---
if canvas_result.json_data is not None:
    new_json = canvas_result.json_data
    
    # 1. Ensure Background matches current image
    if not new_json.get("backgroundImage"):
         bg_config = get_background_config(pil_image, canvas_width, canvas_height)
         new_json["backgroundImage"] = bg_config
    
    # 2. Extract ROI for Analysis and ENFORCE Single ROI visually
    current_rois_extracted = []
    objects_list = new_json.get("objects", [])
    
    if objects_list:
        # Filter for rectangles drawn by user
        rects = [obj for obj in objects_list if obj.get("type") == "rect"]
        
        if rects:
            # Enforce single ROI rule: if multiple exist, keep only the most recent one
            latest_rect = rects[-1]
            
            # If the user drew a new one, strip out the older ones to update the visual state
            if len(rects) > 1:
                new_json["objects"] = [latest_rect]
                st.session_state.drawing_states[selected_name] = new_json
                st.session_state.canvas_init_state = new_json # Update the frozen state
                st.session_state.canvas_key_suffix += 1 # Force canvas frontend remount
                st.rerun() 
                
            left = int(latest_rect["left"] / scale_factor)
            top = int(latest_rect["top"] / scale_factor)
            width = int(latest_rect["width"] / scale_factor)
            height = int(latest_rect["height"] / scale_factor)
            current_rois_extracted.append((left, top, left + width, top + height))

    # 3. Update State Persistence
    st.session_state.drawing_states[selected_name] = new_json
    st.session_state.rois[selected_name] = current_rois_extracted

annotated_count = len([k for k, v in st.session_state.rois.items() if v])
st.caption(f"Annotated **{annotated_count}/{n_images}** images.")

# --- Step 3: Time Configuration ---
st.header("3. Time Configuration")

if "current_timebase" not in st.session_state:
    st.session_state.current_timebase = "Seconds"

new_timebase = st.selectbox("Global Timebase", ["Seconds", "Minutes", "Hours"])

# Sync default time data
if "time_data" not in st.session_state or len(st.session_state.time_data) != n_images:
    default_data = {
        "Filename": loaded_filenames,
        "Time": [i * 10.0 for i in range(n_images)]
    }
    st.session_state.time_data = pd.DataFrame(default_data)

# Handle Timebase Conversion dynamically within the table
if new_timebase != st.session_state.current_timebase:
    df = st.session_state.time_data
    
    # Convert existing values to Seconds first
    if st.session_state.current_timebase == "Minutes":
        df["Time"] *= 60.0
    elif st.session_state.current_timebase == "Hours":
        df["Time"] *= 3600.0
        
    # Convert Seconds to the new selected timebase
    if new_timebase == "Minutes":
        df["Time"] /= 60.0
    elif new_timebase == "Hours":
        df["Time"] /= 3600.0
        
    st.session_state.current_timebase = new_timebase
    st.session_state.time_data = df
    st.rerun()

# Display the Data Editor with dynamic column naming
display_df = st.session_state.time_data.copy()
time_col_name = f"Time ({new_timebase})"
display_df.rename(columns={"Time": time_col_name}, inplace=True)

edited_df = st.data_editor(display_df, use_container_width=True)

# Save user edits back to the internal state
st.session_state.time_data["Time"] = edited_df[time_col_name]

# --- Step 4: Analysis & Visualization ---
st.header("4. Analysis & Output")

# Analysis Settings
with st.expander("Analysis & Plot Settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        color_space = st.radio("Color Space", ["RGB", "HSV"], horizontal=True)
    with c2:
        plot_type = st.radio("Plot Type", ["Bar Chart", "Line Chart"], horizontal=True)
    with c3:
        show_legend = st.checkbox("Show Legend", value=True)
        show_grid = st.checkbox("Show Grid", value=True)

# Check if we need to auto-recalculate
if "last_color_space" not in st.session_state:
    st.session_state.last_color_space = color_space

# If color space changed and we have results, force rerun
auto_rerun = False
if color_space != st.session_state.last_color_space:
    if "results" in st.session_state:
        auto_rerun = True
    st.session_state.last_color_space = color_space

def perform_analysis():
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(loaded_filenames):
        status_text.text(f"Processing {filename}...")
        
        img_bgr = image_cache[filename]['bgr']
        img_h, img_w, _ = img_bgr.shape
        
        # Get Time directly from the edited dataframe
        row = edited_df[edited_df["Filename"] == filename]
        if row.empty:
            continue
        row = row.iloc[0]
            
        t_val = float(row[time_col_name])
            
        frame_data = {
            "Filename": filename,
            "Time": t_val
        }
        
        # Get ROI
        file_rois = st.session_state.rois.get(filename, [])
        
        # Convert Color Space
        if color_space == "HSV":
            img_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        else:
            img_proc = img_bgr
            
        val1, val2, val3 = np.nan, np.nan, np.nan
        
        # Process the single ROI if it exists
        if file_rois:
            x1, y1, x2, y2 = file_rois[0]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            region = img_proc[y1:y2, x1:x2]
            
            if region.size > 0:
                mean = cv2.mean(region)
                if color_space == "RGB":
                    val3 = mean[0] / 255.0 * 100 # Blue
                    val2 = mean[1] / 255.0 * 100 # Green
                    val1 = mean[2] / 255.0 * 100 # Red
                else: # HSV
                    val1 = mean[0] * 2.0  # H (0-360)
                    val2 = mean[1] / 255.0 * 100 # S
                    val3 = mean[2] / 255.0 * 100 # V
                    
        frame_data["Val1"] = val1 
        frame_data["Val2"] = val2 
        frame_data["Val3"] = val3 

        results.append(frame_data)
        progress_bar.progress((i + 1) / n_images)
        
    st.session_state.results = pd.DataFrame(results)
    status_text.empty()
    st.success("Analysis Complete!")

if st.button("Run Analysis", type="primary") or auto_rerun:
    perform_analysis()
    if auto_rerun:
        st.rerun()

# --- Step 5: Visualization ---
if "results" in st.session_state:
    res_df = st.session_state.results
    
    # Validate columns to prevent KeyError from stale state
    if "Time" not in res_df.columns:
        del st.session_state.results
        st.rerun()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    res_df = res_df.sort_values("Time")
    t = res_df["Time"]
    
    if color_space == "RGB":
        labels = ["Red", "Green", "Blue"]
        colors = ["#ff4b4b", "#4caf50", "#2196f3"] # Improved colors for visual clarity
        y_label = "Intensity (%)"
    else:
        labels = ["Hue", "Saturation", "Value"]
        colors = ["m", "c", "k"]
        y_label = "Value (deg / %)"
        
    if plot_type == "Bar Chart":
        # Grouped Bar Chart
        x_indexes = np.arange(len(res_df))
        width = 0.25
        
        ax.bar(x_indexes - width, res_df["Val1"], width, color=colors[0], label=labels[0])
        ax.bar(x_indexes,         res_df["Val2"], width, color=colors[1], label=labels[1])
        ax.bar(x_indexes + width, res_df["Val3"], width, color=colors[2], label=labels[2])
        
        # Align ticks with Time values
        ax.set_xticks(x_indexes)
        ax.set_xticklabels([f"{val:.1f}" for val in t], rotation=45)
    else:
        # Line Chart
        ax.plot(t, res_df["Val1"], color=colors[0], linestyle="-", marker='o', label=labels[0], alpha=0.8)
        ax.plot(t, res_df["Val2"], color=colors[1], linestyle="--", marker='s', label=labels[1], alpha=0.8)
        ax.plot(t, res_df["Val3"], color=colors[2], linestyle="-.", marker='^', label=labels[2], alpha=0.8)

    ax.set_xlabel(f"Time ({new_timebase})")
    ax.set_ylabel(y_label)
    ax.set_title(f"{color_space} Analysis over Time")
    
    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Avoid overlapping elements
    plt.tight_layout()
    st.pyplot(fig)
    
    st.header("Data Table")
    
    # Rename columns for final presentation table based on Color Space
    final_display_df = res_df.copy()
    final_display_df.rename(columns={
        "Time": f"Time ({new_timebase})",
        "Val1": labels[0],
        "Val2": labels[1],
        "Val3": labels[2]
    }, inplace=True)
    
    st.dataframe(final_display_df, use_container_width=True)
    
    csv = final_display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "analysis.csv", "text/csv")