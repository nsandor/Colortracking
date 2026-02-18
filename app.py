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
2. **Select** each image and **Draw** Regions of Interest (ROIs).
3. **Name** your ROIs (Optional).
4. **Set Times** for each image.
5. **Analyze** to generate plots and data.
""")

# Initialize Session State
if "rois" not in st.session_state:
    st.session_state.rois = {}  # {filename: [roi1, roi2]}
if "roi_names" not in st.session_state:
    st.session_state.roi_names = [] # ["ROI 1", "ROI 2", ...] (Global list)
if "drawing_states" not in st.session_state:
    st.session_state.drawing_states = {} # {filename: json_dict}

# --- Sidebar: File Upload & Settings ---
st.sidebar.header("1. Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Upload Images", 
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], 
    accept_multiple_files=True
)

st.sidebar.header("Settings")
st.sidebar.info("Upload images to begin analysis.")

# Caching mechanism using cache_resource to keep objects in memory (not pickled)
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
                'pil': pil_img, # Keep PIL object stable
                'bgr': img_bgr, # Keep BGR for cv2 processing
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

# Image Scrubber
if n_images > 1:
    selected_name = st.select_slider(
        "Select Image", 
        options=loaded_filenames, 
        value=loaded_filenames[0],
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

st.caption(f"Drawing on: {selected_name} | Size: {img_w}x{img_h}")

# Tool Selection
col_tool, col_info = st.columns([1, 2])
with col_tool:
    tool_mode = st.radio("Tool Mode", ["Draw ROI", "Edit/Move ROIs"], horizontal=True)

if tool_mode == "Draw ROI":
    drawing_mode = "rect"
else:
    drawing_mode = "transform"

# Copy ROI Buttons
col1, col2, col3 = st.columns([1, 1, 3])
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
                st.toast(f"Copied {len(prev_rois)} ROIs.")
                st.rerun()
            else:
                st.warning("No ROIs in previous.")
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
            st.toast(f"Applied ROIs to {len(loaded_filenames)} images.")
            st.rerun()
        else:
            st.warning("No ROIs to apply.")

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
# We must ONLY update 'initial_drawing' when the image or mode changes.
# If we update it on every drawing action, it causes a rerun loop (flickering).

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
        for (x1, y1, x2, y2) in saved_rois:
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
        # Ensure Background is present (in case of legacy state or issues)
        if not current_state.get("backgroundImage"):
            current_state["backgroundImage"] = get_background_config(pil_image, canvas_width, canvas_height)

    # 2. Update Frozen State
    st.session_state.canvas_init_state = current_state
    
    # 3. Update Checkpoints
    st.session_state.last_selected_name = selected_name
    st.session_state.last_tool_mode = tool_mode

# --- Render Canvas ---
# Use the FROZEN state. Do not pass dynamic updates here.
canvas_key = f"canvas_{selected_name}_{tool_mode}"

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
    
    # 1. Ensure Background matches current image (Fix for 'vanishing' or incorrect BG)
    # The canvas might return a state with a missing or stale background if the client-side wasn't fully ready.
    # We verify the 'backgroundImage' presence.
    if not new_json.get("backgroundImage"):
         bg_config = get_background_config(pil_image, canvas_width, canvas_height)
         new_json["backgroundImage"] = bg_config
    
    # 2. Update State Persistence
    st.session_state.drawing_states[selected_name] = new_json
    
    # 3. Extract ROIs for Analysis
    current_rois_extracted = []
    # Only look at 'objects' list. Background is essentially invisible to this logic now.
    objects_list = new_json.get("objects", [])
    if objects_list:
        df_objects = pd.json_normalize(objects_list)
        if not df_objects.empty:
            for _, obj in df_objects.iterrows():
                if obj["type"] == "rect":
                    left = int(obj["left"] / scale_factor)
                    top = int(obj["top"] / scale_factor)
                    width = int(obj["width"] / scale_factor)
                    height = int(obj["height"] / scale_factor)
                    current_rois_extracted.append((left, top, left + width, top + height))

    st.session_state.rois[selected_name] = current_rois_extracted
    
    # Update Global ROI Names list logic
    current_roi_count = len(current_rois_extracted)
    
    # Ensure global list is long enough
    if len(st.session_state.roi_names) < current_roi_count:
        for k in range(len(st.session_state.roi_names), current_roi_count):
            st.session_state.roi_names.append(f"ROI {k+1}")

# Status & Naming
current_rois = st.session_state.rois.get(selected_name, [])
if current_rois:
    st.subheader("ROI Names (Global)")
    st.info("Names are shared across all images.")
    cols = st.columns(3)
    
    # Only show inputs for the number of ROIs in THIS image
    # Note: We work on the GLOBAL list directly
    for idx in range(len(current_rois)):
        # Safety check if global list is shorter (shouldn't happen due to logic above)
        if idx >= len(st.session_state.roi_names):
             st.session_state.roi_names.append(f"ROI {idx+1}")
             
        col_idx = idx % 3
        with cols[col_idx]:
             st.session_state.roi_names[idx] = st.text_input(
                f"Name for ROI {idx+1}", 
                value=st.session_state.roi_names[idx],
                key=f"roi_name_global_{idx}"
            )

annotated_count = len([k for k, v in st.session_state.rois.items() if v])
st.caption(f"Annotated **{annotated_count}/{n_images}** images.")

# --- Step 3: Time Configuration ---
st.header("3. Time Configuration")

# Sync time data
if "time_data" not in st.session_state or len(st.session_state.time_data) != n_images:
     default_data = {
        "Filename": loaded_filenames,
        "Time": [i * 10.0 for i in range(n_images)],
        "Unit": ["seconds"] * n_images
    }
     st.session_state.time_data = pd.DataFrame(default_data)

edited_df = st.data_editor(st.session_state.time_data, use_container_width=True)
st.session_state.time_data = edited_df

# --- Step 4: Analysis ---
# --- Step 4: Analysis & Visualization ---
st.header("4. Analysis & Output")

# Analysis Settings
with st.expander("Analysis & Plot Settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        color_space = st.radio("Color Space", ["RGB", "HSV"], horizontal=True)
    with c2:
        show_legend = st.checkbox("Show Legend", value=True)
        show_grid = st.checkbox("Show Grid", value=True)
    with c3:
        pass

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
        
        # Get Time
        row = edited_df[edited_df["Filename"] == filename]
        if row.empty:
            continue
        row = row.iloc[0]
            
        t_val = row["Time"]
        u_val = row["Unit"]
        
        # Normalize Time
        t_sec = float(t_val)
        if str(u_val).startswith("min"):
            t_sec *= 60
        elif str(u_val).startswith("h"):
            t_sec *= 3600
            
        frame_data = {
            "Filename": filename,
            "Time (s)": t_sec,
            "Original Time": t_val,
            "Unit": u_val
        }
        
        # Get ROIs & Names
        file_rois = st.session_state.rois.get(filename, [])
        # Use Global Names
        global_roi_names = st.session_state.roi_names
        
        # Ensure names match count (bulletproofing)
        if len(global_roi_names) < len(file_rois):
             # This really shouldn't happen if the UI logic works, but just in case
             # we temporarily extend it for this read (not saving back to state here to avoid side effects in loop)
             extended_names = global_roi_names + [f"ROI {k+1}" for k in range(len(global_roi_names), len(file_rois))]
        else:
             extended_names = global_roi_names
        
        # Convert Color Space
        if color_space == "HSV":
            img_proc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        else:
            img_proc = img_bgr
        
        for r_idx, (x1, y1, x2, y2) in enumerate(file_rois):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            region = img_proc[y1:y2, x1:x2]
            
            roi_name = extended_names[r_idx]
            # Sanitize name for DataFrame column
            safe_name = roi_name.replace(" ", "_").replace(".", "")
            
            val1, val2, val3 = np.nan, np.nan, np.nan
            
            if region.size > 0:
                mean = cv2.mean(region)
                if color_space == "RGB":
                    val3 = mean[0] / 255.0 * 100 # Blue
                    val2 = mean[1] / 255.0 * 100 # Green
                    val1 = mean[2] / 255.0 * 100 # Red
                    
                    frame_data[f"Val1_{safe_name}"] = val1 # Red
                    frame_data[f"Val2_{safe_name}"] = val2 # Green
                    frame_data[f"Val3_{safe_name}"] = val3 # Blue
                else: # HSV
                    h_raw = mean[0]
                    s_raw = mean[1]
                    v_raw = mean[2]
                    
                    val1 = h_raw * 2.0  # H (0-360)
                    val2 = s_raw / 255.0 * 100 # S
                    val3 = v_raw / 255.0 * 100 # V
                    
                    frame_data[f"Val1_{safe_name}"] = val1 # Hue
                    frame_data[f"Val2_{safe_name}"] = val2 # Sat
                    frame_data[f"Val3_{safe_name}"] = val3 # Val

        results.append(frame_data)
        progress_bar.progress((i + 1) / n_images)
        
    st.session_state.results = pd.DataFrame(results)
    status_text.empty()
    st.success("Analysis Complete!")

if st.button("Run Analysis") or auto_rerun:
    perform_analysis()
    if auto_rerun:
        st.rerun()

# --- Step 5: Visualization ---
if "results" in st.session_state:
    res_df = st.session_state.results
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    res_df = res_df.sort_values("Time (s)")
    t = res_df["Time (s)"]
    
    # Determine labels based on current color space setting
    # Note: If the user changes color space but hasn't re-run analysis, 
    # the data might be stale.
    # However, our auto_rerun logic attempts to fix this.
    
    if color_space == "RGB":
        labels = ["Red", "Green", "Blue"]
        colors = ["r", "g", "b"]
        styles = ["--", ":", "-."]
        y_label = "Intensity (%)"
    else:
        labels = ["Hue", "Saturation", "Value"]
        colors = ["m", "c", "k"]
        styles = ["-", "--", ":"]
        y_label = "Value (deg / %)"
        
    area_cols = [c for c in res_df.columns if c.startswith("Val1_")]
    # Extract names from columns: Val1_{Name} -> {Name}
    roi_names_found = [c.replace("Val1_", "") for c in area_cols]
    
    for name in roi_names_found:
        if f"Val1_{name}" in res_df:
            ax.plot(t, res_df[f"Val1_{name}"], color=colors[0], linestyle=styles[0], marker='o', label=f"{labels[0]} {name}", alpha=0.7)
        if f"Val2_{name}" in res_df:
            ax.plot(t, res_df[f"Val2_{name}"], color=colors[1], linestyle=styles[1], marker='s', label=f"{labels[1]} {name}", alpha=0.7)
        if f"Val3_{name}" in res_df:
            ax.plot(t, res_df[f"Val3_{name}"], color=colors[2], linestyle=styles[2], marker='^', label=f"{labels[2]} {name}", alpha=0.7)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{color_space} Analysis over Time")
    
    if show_grid:
        ax.grid(True)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    st.pyplot(fig)
    
    st.header("Data Table")
    st.dataframe(res_df)
    
    csv = res_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "analysis.csv", "text/csv")
