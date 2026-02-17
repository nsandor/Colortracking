import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import (
    filedialog, Tk, Toplevel, Button, Label, Canvas, Frame,
    StringVar, Entry, OptionMenu
)
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


# Area and time selection

def select_areas_for_all_images(root, image_paths):

    n_images = len(image_paths)
    if n_images == 0:
        return [], [], []

    rects_per_image = [[] for _ in range(n_images)]
    rect_ids_per_image = [[] for _ in range(n_images)]
    scale_factors = [(1.0, 1.0)] * n_images

    time_values = [None] * n_images
    unit_values = ["seconds"] * n_images

    win = Toplevel(root)
    win.title("Select Areas and Times")

    # Bring selector window to front
    win.lift()
    win.focus_force()
    win.attributes("-topmost", True)
    win.after_idle(win.attributes, "-topmost", False)

    # LEFT: canvas + status
    left_frame = Frame(win)
    left_frame.pack(side="left", fill="both", expand=True)

    status_label = Label(left_frame, text="", pady=5)
    status_label.pack(side="top")

    canvas = Canvas(left_frame, bg="black")
    canvas.pack(side="top", fill="both", expand=True)

    # RIGHT: time + controls
    right_frame = Frame(win)
    right_frame.pack(side="right", fill="y", padx=10, pady=10)


    time_frame = Frame(right_frame)
    time_frame.pack(side="top", pady=10)

    Label(time_frame, text="Time:").pack()
    time_entry_var = StringVar()
    time_entry = Entry(
    time_frame,
    textvariable=time_entry_var,
    width=10,
    highlightthickness=2,
    highlightbackground="black",   # outline when not focused
    highlightcolor="black"         # outline when focused
)

    time_entry.pack()

    Label(time_frame, text="Units:").pack()
    unit_var = StringVar(value="seconds")
    unit_menu = OptionMenu(time_frame, unit_var, "seconds", "minutes", "hours")
    unit_menu.pack()

    current_idx = 0
    tkimg_cache = [None] * n_images
    max_display_width = 900

    drawing = False
    start_x = start_y = 0
    current_rect_id = None

    # ---- ERROR POPUP ----
    def show_error(title, msg):
        messagebox.showerror(title, msg)
        status_label.config(text=msg)

    def update_status(idx):
        txt = f"Image {idx+1}/{n_images}\nAreas: {len(rects_per_image[idx])}"
        if time_values[idx] is not None:
            txt += f"\nTime: {time_values[idx]} {unit_values[idx]}"
        else:
            txt += "\nTime: NOT SET"
        status_label.config(text=txt)

    def load_image(idx):
        img = cv2.imread(image_paths[idx])
        if img is None:
            show_error("Image Error", "Could not load image.")
            return

        h, w = img.shape[:2]
        if w > max_display_width:
            s = max_display_width / w
            dw, dh = max_display_width, int(h * s)
        else:
            dw, dh = w, h

        sx = w / float(dw)
        sy = h / float(dh)
        scale_factors[idx] = (sx, sy)

        resized = cv2.resize(img, (dw, dh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        tkimg_cache[idx] = tkimg

        canvas.delete("all")
        canvas.config(width=dw, height=dh)
        canvas.create_image(0, 0, anchor="nw", image=tkimg)

        rect_ids_per_image[idx] = []
        for (x1, y1, x2, y2) in rects_per_image[idx]:
            dx1 = x1 / sx
            dy1 = y1 / sy
            dx2 = x2 / sx
            dy2 = y2 / sy
            rid = canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="green", width=2)
            rect_ids_per_image[idx].append(rid)

        if time_values[idx] is not None:
            time_entry_var.set(str(time_values[idx]))
            unit_var.set(unit_values[idx])
        else:
            time_entry_var.set("")
            unit_var.set("seconds")

        update_status(idx)

        # Enable/disable Prev button
        if idx == 0:
            prev_btn.config(state="disabled")
        else:
            prev_btn.config(state="normal")

        # Enable/disable Next button and show/hide Done on last image
        if idx == n_images - 1:
            next_btn.config(state="disabled")
            done_btn.pack(pady=5)  # Show Done ONLY on last image
        else:
            done_btn.pack_forget()
            next_btn.config(state="normal")

    def save_time_for_current():
        idx = current_idx
        txt = time_entry_var.get().strip()
        if not txt:
            show_error("Time Error", "Please enter a time and click Save Time.")
            return
        try:
            val = float(txt)
        except ValueError:
            show_error("Time Error", "Time must be a number.")
            return

        time_values[idx] = val
        unit_values[idx] = unit_var.get()
        update_status(idx)

    Button(time_frame, text="Save Time", command=save_time_for_current).pack(pady=5)

    # Buttons
    controls = Frame(right_frame)
    controls.pack(side="top", pady=10)

    prev_btn = Button(controls, text="Previous Image", width=15)
    next_btn = Button(controls, text="Next Image", width=15)
    undo_btn = Button(controls, text="Undo Last Area", width=15)
    done_btn = Button(controls, text="Done", width=15)  # HIDDEN until last image

    prev_btn.pack(pady=5)
    next_btn.pack(pady=5)
    undo_btn.pack(pady=5)
    # done_btn only appears on last image

    # Instructions
    instructions = (
        "Instructions:\n"
        "• Click and drag on the image to draw a green box.\n"
        "• Enter the time, choose units from the dropdown menu, then click 'Save Time'.\n"
        "• Do this for EACH image.\n"
        "• On the last image, click Done to continue."
    )
    Label(right_frame, text=instructions, wraplength=260, justify="left").pack(pady=10)

    # Button functions
    def next_image():
        nonlocal current_idx

        if len(rects_per_image[current_idx]) == 0:
            show_error("Selection Error", "Please select at least one area before continuing.")
            return

        if time_values[current_idx] is None:
            show_error("Time Error", "Please enter and save a time before continuing.")
            return

        if current_idx < n_images - 1:
            current_idx += 1
            load_image(current_idx)

    def prev_image():
        nonlocal current_idx
        if current_idx > 0:
            current_idx -= 1
            load_image(current_idx)

    def undo_last():
        idx = current_idx
        if rects_per_image[idx]:
            rects_per_image[idx].pop()
        if rect_ids_per_image[idx]:
            canvas.delete(rect_ids_per_image[idx].pop())
        update_status(idx)

    def on_done():
        missing_area = [i for i, lst in enumerate(rects_per_image) if len(lst) == 0]
        missing_time = [i for i, t in enumerate(time_values) if t is None]

        if missing_area:
            show_error("Selection Error", f"Missing areas on images: {', '.join(str(i+1) for i in missing_area)}")
            return
        if missing_time:
            show_error("Time Error", f"Missing times on images: {', '.join(str(i+1) for i in missing_time)}")
            return

        win.destroy()

    prev_btn.config(command=prev_image)
    next_btn.config(command=next_image)
    undo_btn.config(command=undo_last)
    done_btn.config(command=on_done)

    # Mouse callbacks
    def mouse_down(event):
        nonlocal drawing, start_x, start_y, current_rect_id
        drawing = True
        start_x, start_y = event.x, event.y
        current_rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y,
                                                  outline="green", width=2)

    def mouse_move(event):
        if drawing and current_rect_id is not None:
            canvas.coords(current_rect_id, start_x, start_y, event.x, event.y)

    def mouse_up(event):
        nonlocal drawing, current_rect_id
        if not drawing or current_rect_id is None:
            return

        drawing = False

        x1, y1 = min(start_x, event.x), min(start_y, event.y)
        x2, y2 = max(start_x, event.x), max(start_y, event.y)

        if x2 - x1 < 3 or y2 - y1 < 3:
            canvas.delete(current_rect_id)
            current_rect_id = None
            return

        sx, sy = scale_factors[current_idx]
        ox1, oy1 = int(x1 * sx), int(y1 * sy)
        ox2, oy2 = int(x2 * sx), int(y2 * sy)

        rects_per_image[current_idx].append((ox1, oy1, ox2, oy2))
        rect_ids_per_image[current_idx].append(current_rect_id)
        update_status(current_idx)

        current_rect_id = None

    canvas.bind("<ButtonPress-1>", mouse_down)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_up)

    load_image(0)

    win.grab_set()
    root.wait_window(win)

    return rects_per_image, time_values, unit_values


# Image processing

def process_images(image_paths, rects, times, units):
    n_images = len(image_paths)
    max_areas = max(len(a) for a in rects)

    rgb = {"time": [], "time_label": []}

    for a in range(max_areas):
        rgb[f"R_area_{a}"] = []
        rgb[f"G_area_{a}"] = []
        rgb[f"B_area_{a}"] = []

   # Choose taregt unit
    unit_counts = {"seconds": 0, "minutes": 0, "hours": 0}
    for u in units:
        if u.startswith("min"):
            unit_counts["minutes"] += 1
        elif u.startswith("h"):
            unit_counts["hours"] += 1
        else:
            unit_counts["seconds"] += 1

    target_unit = max(unit_counts, key=unit_counts.get)
    rgb["_time_unit"] = target_unit  # store for later axis labelling

    # Process the images at each time
    for i, path in enumerate(image_paths):
        t_raw = times[i]
        u = units[i]

        # Convert incoming time to seconds first
        t_sec = t_raw
        if u.startswith("min"):
            t_sec *= 60
        elif u.startswith("h"):
            t_sec *= 3600

        # Now convert seconds to target unit
        if target_unit == "minutes":
            t_final = t_sec / 60.0
        elif target_unit == "hours":
            t_final = t_sec / 3600.0
        else:
            t_final = t_sec

        rgb["time"].append(t_final)
        rgb["time_label"].append(f"{t_final:.2f} {target_unit}")

        img = cv2.imread(path)
        if img is None:
            for a in range(max_areas):
                rgb[f"R_area_{a}"].append(np.nan)
                rgb[f"G_area_{a}"].append(np.nan)
                rgb[f"B_area_{a}"].append(np.nan)
            continue

        for a in range(max_areas):
            if a < len(rects[i]):
                x1, y1, x2, y2 = rects[i][a]
                region = img[y1:y2, x1:x2]
                if region.size == 0:
                    R = G = B = np.nan
                else:
                    mean = cv2.mean(region)
                    B = mean[0] / 200 * 100
                    G = mean[1] / 200 * 100
                    R = mean[2] / 200 * 100
            else:
                R = G = B = np.nan

            rgb[f"R_area_{a}"].append(R)
            rgb[f"G_area_{a}"].append(G)
            rgb[f"B_area_{a}"].append(B)

    return rgb


# Graph viewer

def viewer_all(root, rgb, rects_per_image):

    max_areas = max(len(lst) for lst in rects_per_image)
    if max_areas == 0:
        return

    times = np.array(rgb["time"])
    order = np.argsort(times)
    times_sorted = times[order]
    labels_sorted = np.array(rgb["time_label"])[order]
    time_unit = rgb.get("_time_unit", "seconds")

    N = len(times_sorted)
    TOTAL_PAGES = N + 1

    win = Toplevel(root)
    win.title("RGB Analysis Viewer")

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack()

    button_frame = Frame(win)
    button_frame.pack(side="bottom", pady=10)

    prev_btn = Button(button_frame, text="Previous")
    next_btn = Button(button_frame, text="Next")
    save_btn = Button(button_frame, text="Save PNG")
    done_btn = Button(button_frame, text="Done")

    prev_btn.pack(side="left", padx=10)
    next_btn.pack(side="left", padx=10)
    save_btn.pack(side="left", padx=10)
    done_btn.pack(side="left", padx=10)

    idx = 0

    h_R = "//"
    h_G = "\\\\"
    h_B = "xx"

    line_style_R = ("red", "--", "o")
    line_style_G = ("green", ":", "s")
    line_style_B = ("blue", "-.", "^")

    def draw_page():
        ax.clear()

        if idx < N:
            # BAR GRAPH
            R_vals, G_vals, B_vals = [], [], []
            for a in range(max_areas):
                R_vals.append(rgb[f"R_area_{a}"][order[idx]])
                G_vals.append(rgb[f"G_area_{a}"][order[idx]])
                B_vals.append(rgb[f"B_area_{a}"][order[idx]])

            R = np.nanmean(R_vals)
            G = np.nanmean(G_vals)
            B = np.nanmean(B_vals)

            x = np.array([0, 1, 2])
            w = 0.6

            ax.bar(0, R, w, color="red", hatch=h_R)
            ax.bar(1, G, w, color="green", hatch=h_G)
            ax.bar(2, B, w, color="blue", hatch=h_B)

            ax.set_xticks(x)
            ax.set_xticklabels(["Red", "Green", "Blue"])
            ax.set_ylabel("Colour Intensity (%)")
            ax.set_title(f"RGB % — {labels_sorted[idx]}")
            ax.grid(alpha=0.3)

        else:
            # LINE GRAPH
            for a in range(max_areas):
                R = np.array(rgb[f"R_area_{a}"])[order]
                G = np.array(rgb[f"G_area_{a}"])[order]
                B = np.array(rgb[f"B_area_{a}"])[order]

                ax.plot(times_sorted, R,
                        color=line_style_R[0], linestyle=line_style_R[1], marker=line_style_R[2],
                        label="Red")
                ax.plot(times_sorted, G,
                        color=line_style_G[0], linestyle=line_style_G[1], marker=line_style_G[2],
                        label="Green")
                ax.plot(times_sorted, B,
                        color=line_style_B[0], linestyle=line_style_B[1], marker=line_style_B[2],
                        label="Blue")

            ax.set_xlabel(f"Time ({time_unit})")
            ax.set_ylabel("Colour Intensity (%)")
            ax.set_title("RGB % vs Time")
            ax.grid(True)
            ax.legend(ncol=2)

        fig.tight_layout()
        canvas.draw()

        # Button visibility
        prev_btn.config(state="normal" if idx > 0 else "disabled")
        next_btn.config(state="normal" if idx < TOTAL_PAGES - 1 else "disabled")

    def confirm_done():
        resp = messagebox.askyesno(
            "Confirm Done",
            "By clicking Done you cannot go back.\n\nHave you saved everything you need?"
        )
        if resp:
            win.destroy()

    def next_page():
        nonlocal idx
        if idx < TOTAL_PAGES - 1:
            idx += 1
            draw_page()

    def prev_page():
        nonlocal idx
        if idx > 0:
            idx -= 1
            draw_page()

    def save_png():
        f = filedialog.asksaveasfilename(defaultextension=".png")
        if f:
            fig.savefig(f, dpi=300)

    prev_btn.config(command=prev_page)
    next_btn.config(command=next_page)
    save_btn.config(command=save_png)
    done_btn.config(command=confirm_done)

    draw_page()


# Main logic

if __name__ == "__main__":
    root = Tk()
    root.withdraw()   # keep root invisible

    # Bring messagebox to the front
    root.attributes("-topmost", True)
    messagebox.showinfo(
        "Select Images",
        "Please select your image files.\n\n"
        "Accepted file types:\n"
        "• PNG\n• JPG / JPEG\n• BMP\n• TIF / TIFF\n\n"
        "Only these file types will be selectable."
    )
    root.after_idle(root.attributes, "-topmost", False)

    # Proper filtered file dialog
    image_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("Bitmap", "*.bmp"),
            ("TIFF", "*.tif *.tiff")
        ]
    )

    if not image_paths:
        messagebox.showerror("No Files Selected",
                             "You must select at least one valid image file.")
        root.destroy()
        raise SystemExit

    # Continue as normal
    rects, times, units = select_areas_for_all_images(root, image_paths)
    rgb = process_images(image_paths, rects, times, units)
    viewer_all(root, rgb, rects)

    root.mainloop()

    # I hope this is understandable, if you have any questions just shoot me an email :) -Katlyn
