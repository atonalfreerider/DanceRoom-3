import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime
import subprocess
from video_meta import extract_video_metadata
from pathlib import Path
import argparse  # Add this import

class DanceVideoProcessor:
    def __init__(self, root, source_dir=None, output_dir=None):  # Add parameters
        self.root = root
        self.root.title("Dance Video Processor")
        self.videos_info = {}  # Technical info for each video
        self.video_options = {}  # Initialize video options dict
        self.current_video = None
        self.collective_metadata = {  # Shared metadata for all videos
            "lead_dancer": "",
            "follow_dancer": "",
            "event_title": "",
            "dance_level": "novice",
            "dance_type": "Social",
            "date": datetime.now().strftime("%d/%m/%Y")
        }
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Source directory selection
        ttk.Label(main_frame, text="Video Source Directory:").grid(row=0, column=0, sticky=tk.W)
        self.source_dir = tk.StringVar(value=source_dir or "")  # Initialize with arg if provided
        ttk.Entry(main_frame, textvariable=self.source_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=0, column=2)
        ttk.Button(main_frame, text="Load", command=self.load_videos).grid(row=0, column=3)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar(value=output_dir or "")  # Initialize with arg if provided
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        # Create a frame for video list and options
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky=(tk.W, tk.E))
        
        # Left side: Video list
        list_frame = ttk.Frame(video_frame)
        list_frame.grid(row=0, column=0, padx=5)
        
        ttk.Label(list_frame, text="Videos:").grid(row=0, column=0, sticky=tk.W)
        self.video_listbox = tk.Listbox(list_frame, width=50, height=5)
        self.video_listbox.grid(row=1, column=0, pady=5)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        
        # Right side: Per-video options
        self.options_frame = ttk.LabelFrame(video_frame, text="Video Processing Options")
        self.options_frame.grid(row=0, column=1, padx=5, sticky=(tk.N, tk.S))
        
        # Video options checkboxes with save bindings
        self.checkbox_vars = {}
        self.default_options = {
            "fixed_focal_length": True,
            "translating_position": True,
            "changing_orientation": True
        }
        
        checkboxes = [
            ("fixed_focal_length", "Fixed Focal Length"),
            ("translating_position", "Translating Position"),
            ("changing_orientation", "Changing Orientation")
        ]
        
        # Initialize video_options
        self.video_options = {}
        
        # Create checkboxes with modified binding
        for i, (option_id, label) in enumerate(checkboxes):
            var = tk.BooleanVar(value=True)
            checkbox = ttk.Checkbutton(
                self.options_frame,
                text=label,
                variable=var,
                command=lambda opt=option_id: self.on_checkbox_changed(opt)
            )
            checkbox.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            self.checkbox_vars[option_id] = var

        # Collective metadata form
        form_frame = ttk.LabelFrame(main_frame, text="Collection Metadata", padding="5")
        form_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Form fields
        self.form_fields = {}
        fields = [
            ("Lead Dancer:", "lead_dancer"),
            ("Follow Dancer:", "follow_dancer"),
            ("Event Title:", "event_title"),
            ("Dance Level:", "dance_level", ["novice", "intermediate", "advanced", "pro"]),
            ("Dance Type:", "dance_type", ["JnJ", "Demo", "Lesson", "Social"]),
            ("Date:", "date"),
        ]
        
        for i, field in enumerate(fields):
            ttk.Label(form_frame, text=field[0]).grid(row=i, column=0, sticky=tk.W)
            if len(field) == 3:  # Combobox
                var = tk.StringVar()
                widget = ttk.Combobox(form_frame, textvariable=var, values=field[2], state='readonly')
                widget.grid(row=i, column=1, sticky=(tk.W, tk.E))
                self.form_fields[field[1]] = var
            else:  # Entry
                var = tk.StringVar()
                widget = ttk.Entry(form_frame, textvariable=var)
                widget.grid(row=i, column=1, sticky=(tk.W, tk.E))
                self.form_fields[field[1]] = var
                if field[1] == "date":
                    var.set(datetime.now().strftime("%d/%m/%Y"))
            
            # Bind all form fields to save on change
            if isinstance(widget, ttk.Entry):
                widget.bind('<KeyRelease>', lambda e, f=field[1]: self.on_field_change(e, f))
            else:  # Combobox
                widget.bind('<<ComboboxSelected>>', lambda e, f=field[1]: self.on_field_change(e, f))

        # Process button frame
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=4, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Add process button
        self.process_button = ttk.Button(
            process_frame, 
            text="PROCESS ALL VIDEOS", 
            command=self.process_videos,
            style='Process.TButton'
        )
        self.process_button.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Configure process button style
        style = ttk.Style()
        style.configure('Process.TButton', font=('helvetica', 12, 'bold'))

        # Load cached metadata
        self.load_cached_metadata()
        if source_dir:
            self.load_videos()

    def update_video_list_display(self):
        self.video_listbox.delete(0, tk.END)
        for video, info in self.videos_info.items():
            # Calculate FPS from frame count and duration if both exist
            fps = "N/A"
            if "frame_count" in info and "duration" in info and info["duration"] > 0:
                fps = f"{info['frame_count'] / info['duration']:.2f}"
            duration = info.get("duration", "N/A")
            display_text = f"{video} (FPS: {fps}, Duration: {duration:.2f}s)"
            self.video_listbox.insert(tk.END, display_text)

    def load_videos(self):
        directory = self.source_dir.get()
        if not directory:
            messagebox.showerror("Error", "Please select a source directory")
            return
        
        try:
            json_path = os.path.join(directory, "videos_metadata.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if "collective_metadata" in data:
                        stored_metadata = data["collective_metadata"]
                        self.collective_metadata = stored_metadata
                        # Update all form fields from stored metadata
                        for field, var in self.form_fields.items():
                            if field in stored_metadata:
                                var.set(stored_metadata[field])
                    
                    self.videos_info = data.get("videos_info", {})
                    
                    # Convert stored options to BooleanVar
                    stored_options = data.get("video_options", {})
                    self.video_options.clear()
                    for video, options in stored_options.items():
                        self.video_options[video] = {
                            opt: tk.BooleanVar(value=val)
                            for opt, val in options.items()
                        }
            
            # Only scan for new videos, don't overwrite existing metadata
            for file in os.listdir(directory):
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    # Only process new videos
                    if file not in self.videos_info:
                        video_path = os.path.join(directory, file)
                        try:
                            tech_info = extract_video_metadata(video_path)
                            self.videos_info[file] = tech_info
                        except Exception as e:
                            print(f"Error extracting metadata for {file}: {str(e)}")
                    
                    # Initialize options only for new videos
                    if file not in self.video_options:
                        self.video_options[file] = {
                            opt: tk.BooleanVar(value=self.default_options[opt])
                            for opt in self.default_options
                        }
            
            self.update_video_list_display()
            
            # Select first video if none selected
            if not self.current_video and self.video_listbox.size() > 0:
                self.video_listbox.select_set(0)
                self.video_listbox.event_generate('<<ListboxSelect>>')
            
        except Exception as e:
            print(f"Error in load_videos: {str(e)}")
            # Don't reset data on error
            if not self.videos_info:
                self.videos_info = {}
            if not self.video_options:
                self.video_options = {}

    def on_video_select(self, event):
        selection = self.video_listbox.curselection()
        if selection:
            video_line = self.video_listbox.get(selection[0])
            video_name = video_line.split(" (")[0]  # Extract video name from display text
            self.current_video = video_name
            
            # Update checkboxes with video-specific options
            if video_name in self.video_options:
                for opt_name, checkbox_var in self.checkbox_vars.items():
                    video_opt = self.video_options[video_name][opt_name]
                    checkbox_var.set(video_opt.get())

    def on_checkbox_changed(self, option_id):
        """Handle checkbox state changes"""
        if self.current_video and self.current_video in self.video_options:
            # Update the video_options with the new checkbox state
            self.video_options[self.current_video][option_id].set(
                self.checkbox_vars[option_id].get()
            )
            self.save_metadata()

    def load_cached_metadata(self):
        directory = self.source_dir.get()
        if directory:
            json_path = os.path.join(directory, "videos_metadata.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if "collective_metadata" in data:
                            stored_metadata = data["collective_metadata"]
                            # Only update empty fields
                            for field, var in self.form_fields.items():
                                if not var.get():  # Only update if current value is empty
                                    var.set(stored_metadata.get(field, var.get()))
                            # Update internal metadata without overwriting existing values
                            self.collective_metadata.update({
                                k: v for k, v in stored_metadata.items()
                                if k not in self.collective_metadata or not self.collective_metadata[k]
                            })
                        if "videos_info" in data:
                            self.videos_info = data["videos_info"]
                            self.update_video_list_display()
                        # Load video options if they exist
                        if "video_options" in data:
                            self.video_options = {
                                video: {
                                    opt: tk.BooleanVar(value=val)
                                    for opt, val in options.items()
                                }
                                for video, options in data["video_options"].items()
                            }
                except Exception as e:
                    print(f"Error loading metadata cache: {str(e)}")

    def on_field_change(self, event, field_name):
        """Handle changes to form fields"""
        new_value = event.widget.get()
        self.collective_metadata[field_name] = new_value
        self.save_metadata()

    def save_metadata(self, event=None):
        """Save all metadata including per-video options"""
        directory = self.source_dir.get()
        if directory:
            try:
                json_path = os.path.join(directory, "videos_metadata.json")
                
                # Always save current form field values
                self.collective_metadata.update({
                    field: var.get()
                    for field, var in self.form_fields.items()
                })
                
                # Convert BooleanVar to regular bool for JSON
                video_options_json = {}
                for video, options in self.video_options.items():
                    video_options_json[video] = {
                        opt: var.get() if hasattr(var, 'get') else var
                        for opt, var in options.items()
                    }
                
                with open(json_path, 'w') as f:
                    json.dump({
                        "collective_metadata": self.collective_metadata,
                        "videos_info": self.videos_info,
                        "video_options": video_options_json
                    }, f, indent=2)
            except Exception as e:
                print(f"Error saving metadata: {str(e)}")

    def save_video_options(self, option_changed=None):
        """Save options for currently selected video"""
        if hasattr(self, 'current_video') and self.current_video:
            # Make sure video options exist for current video
            if self.current_video not in self.video_options:
                self.video_options[self.current_video] = {}
            
            # Update all options from checkboxes
            for opt_name, var in self.checkbox_vars.items():
                if self.current_video in self.video_options:
                    # Create new BooleanVar if needed
                    if not isinstance(self.video_options[self.current_video].get(opt_name), tk.BooleanVar):
                        self.video_options[self.current_video][opt_name] = tk.BooleanVar()
                    self.video_options[self.current_video][opt_name].set(var.get())
            
            # Save all metadata
            self.save_metadata()

    def browse_source(self):
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir.set(directory)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def process_videos(self):
        if not self.source_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both source and output directories")
            return
            
        script_path = os.path.join(os.path.dirname(__file__), "process_video.sh")
        if not os.path.exists(script_path):
            messagebox.showerror("Error", "Cannot find process_video.sh script")
            return

        # Process each video
        for video in self.videos_info.keys():
            video_path = os.path.join(self.source_dir.get(), video)
            if os.path.exists(video_path):
                try:
                    # Create sanitized video name for output directory
                    video_name = Path(video).stem.replace(' ', '_')
                    output_subdir = os.path.join(self.output_dir.get(), video_name)
                    
                    # Create temporary symlink with underscored name if original has spaces
                    if ' ' in video:
                        temp_video_path = os.path.join(
                            os.path.dirname(video_path),
                            video_name + Path(video).suffix
                        )
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        os.symlink(video_path, temp_video_path)
                        video_path = temp_video_path
                    
                    # Disable process button during processing
                    self.process_button.configure(state='disabled')
                    self.root.update()
                    
                    try:
                        # Create base command
                        cmd = ["bash", script_path]
                        
                        # Add video options as a single argument
                        options = []
                        if self.video_options[video]["fixed_focal_length"].get():
                            options.append("fixed-focal")
                        if self.video_options[video]["translating_position"].get():
                            options.append("translating")
                        if self.video_options[video]["changing_orientation"].get():
                            options.append("orientation")
                        
                        # Add options as a comma-separated string if any exist
                        if options:
                            cmd.append("--options=" + ",".join(options))
                            
                        # Add input and output paths
                        cmd.extend([video_path, output_subdir])
                        
                        subprocess.run(cmd, check=True)
                        messagebox.showinfo("Success", f"Processed {video}")
                    finally:
                        # Clean up temporary symlink if created
                        if ' ' in video and os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                        # Re-enable process button
                        self.process_button.configure(state='normal')
                        
                except subprocess.CalledProcessError as e:
                    messagebox.showerror("Error", f"Error processing {video}: {str(e)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Unexpected error processing {video}: {str(e)}")
            else:
                messagebox.showerror("Error", f"Video file not found: {video}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dance Video Processor GUI")
    parser.add_argument("--source_dir", help="Directory containing source videos")
    parser.add_argument("--output_dir", help="Directory for processed outputs")
    args = parser.parse_args()

    # Create GUI
    root = tk.Tk()
    app = DanceVideoProcessor(
        root,
        source_dir=args.source_dir,
        output_dir=args.output_dir
    )
    root.mainloop()

if __name__ == "__main__":
    main()
