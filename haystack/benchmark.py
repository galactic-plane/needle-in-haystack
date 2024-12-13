import os
import time
import platform
import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetFanSpeed,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
    NVML_TEMPERATURE_GPU,
)
from alive_progress import alive_bar
from gradio_client import Client, handle_file
import pandas as pd
import signal
import sys
from prettytable import PrettyTable
import threading

# Initialize Gradio client
client = Client("http://127.0.0.1:7860/")

# Florence models to benchmark
models = [
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-base",
]

# Directory containing benchmark images
benchmark_dir = "./benchmark"
resolutions = ["720p", "1080p", "1440p", "4k"]  # Four screen resolutions

# Ensure the benchmark directory exists
if not os.path.exists(benchmark_dir):
    raise FileNotFoundError(f"The directory {benchmark_dir} does not exist.")

# Collect subfolders corresponding to resolutions
resolution_folders = [
    os.path.join(benchmark_dir, res)
    for res in resolutions
    if os.path.isdir(os.path.join(benchmark_dir, res))
]

if not resolution_folders:
    raise FileNotFoundError(f"No subfolders found in {benchmark_dir}.")

# Supported image formats
supported_formats = (".jpg", ".jpeg", ".png", ".webp")


# Handle Ctrl+C to exit gracefully
def signal_handler(sig, frame):
    print("\nBenchmark interrupted. Exiting...")
    nvmlShutdown()  # Shut down NVML
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Display system specs
def display_system_specs():
    print("System Specifications:")
    print(f"  OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Processor: {platform.processor()}")
    print(
        f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )
    memory = psutil.virtual_memory()
    print(f"  Total Memory: {round(memory.total / (1024 ** 3), 2)} GB")
    print()
    print("Press Ctrl+C at any time to stop the benchmark.")
    print()


display_system_specs()

# Initialize NVML for GPU stats
nvmlInit()


# Function to get system stats and track max values
def get_system_stats_tracker():
    stats = {
        "CPU Load": 0,
        "GPU Load": 0,
        "GPU Temp": 0,
        "GPU Core Clock": 0,
        "GPU Fan Speed": "N/A",
        "GPU Power Usage": "N/A",
    }

    def get_stats():
        nonlocal stats

        # CPU Stats
        cpu_load = psutil.cpu_percent(interval=None)
        stats["CPU Load"] = f"CPU Load: {cpu_load}%"       

        # GPU Stats
        try:
            handle = nvmlDeviceGetHandleByIndex(0)  # Use the first GPU

            utilization = nvmlDeviceGetUtilizationRates(handle)
            stats["GPU Load"] = f"{utilization.gpu}%"

            gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            stats["GPU Temp"] = f"GPU Temperature: {gpu_temp}°C"

            gpu_core_clock = nvmlDeviceGetClockInfo(handle, 0)  # Graphics clock
            stats["GPU Core Clock"] = f"GPU Core Clock: {gpu_core_clock} MHz"

            gpu_fan_speed = nvmlDeviceGetFanSpeed(handle)
            stats["GPU Fan Speed"] = f"{gpu_fan_speed}%"

            gpu_power_usage = nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            stats["GPU Power Usage"] = f"{gpu_power_usage} W"
        except Exception:
            pass

        return stats

    return get_stats


# Initialize the stats tracker
get_system_stats = get_system_stats_tracker()


def update_stats_realtime():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        stats = get_system_stats()
        table = PrettyTable()
        table.align = "r"
        table.padding_width = 2
        table.field_names = stats.keys()
        table.add_row(stats.values())
        print(table)
        time.sleep(5)


# Start a thread for real-time stats
threading.Thread(target=update_stats_realtime, daemon=True).start()


# Function to determine optimal batch size
def get_optimal_batch_size(image_resolution, max_memory_fraction=0.8):
    resolution_to_memory = {
        "720p": 50 * 1024 * 1024,  # ~50 MB per image
        "1080p": 100 * 1024 * 1024,  # ~100 MB per image
        "1440p": 200 * 1024 * 1024,  # ~200 MB per image
        "4k": 400 * 1024 * 1024,  # ~400 MB per image
    }
    memory_per_image = resolution_to_memory.get(image_resolution, 100 * 1024 * 1024)

    # Get GPU memory
    handle = nvmlDeviceGetHandleByIndex(0)  # First GPU
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    free_memory = (
        memory_info.free * max_memory_fraction
    )  # Use a fraction of available memory

    # Calculate optimal batch size
    return max(1, int(free_memory / memory_per_image))


# Function to process a batch of images with a model
def process_images_with_model(image_paths, model_id):
    start_time = time.time()
    results = [
        client.predict(
            image=handle_file(image_path),
            task_prompt="Object Detection",
            text_input=None,
            model_id=model_id,
            api_name="/process_image",
        )
        for image_path in image_paths
    ]
    end_time = time.time()
    return end_time - start_time, results


# Benchmarking
summary = []
total_steps = len(models) * sum(
    len(os.listdir(folder)) for folder in resolution_folders if os.listdir(folder)
)

try:
    with alive_bar(total_steps, title="Benchmarking Models", length=20, bar='blocks', force_tty=True) as bar:
        for model in models:
            for folder in resolution_folders:
                folder_name = os.path.basename(folder)
                images = [
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.lower().endswith(supported_formats)
                ]

                if not images:
                    continue

                batch_size = get_optimal_batch_size(folder_name)
                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    bar.text(
                        f"Processing batch ({len(batch)} images) at {folder_name} with {model}"
                    )

                    processing_time, _ = process_images_with_model(batch, model)
                    summary.extend(
                        {
                            "Model": model,
                            "Resolution": folder_name,
                            "Image": os.path.basename(image),
                            "Time (s)": round(processing_time / len(batch), 2),
                            **get_system_stats(),
                        }
                        for image in batch
                    )
                    bar(len(batch))  # Advance the progress bar by the batch size
except KeyboardInterrupt:
    print("\nBenchmark interrupted by user. Exiting...")
    nvmlShutdown()
    sys.exit(0)

# Shut down NVML after the benchmark
nvmlShutdown()

# Print the summary in tabular format
df_summary = pd.DataFrame(summary)
print("\nBenchmark Results:")
if not df_summary.empty:
    grouped = df_summary.groupby(["Model", "Resolution"]).agg(
        {"Time (s)": ["mean", "sum"]}
    )
    print(grouped)

    # Calculate overall benchmark metrics
    avg_times = grouped["Time (s)"]["mean"].groupby("Model").mean()
    best_model = avg_times.idxmin()
    best_model_time = avg_times.min()

    avg_resolution = grouped["Time (s)"]["mean"].groupby("Resolution").mean()
    best_resolution = avg_resolution.idxmin()
    best_resolution_time = avg_resolution.min()

    print("\nOverall Benchmark Results:")
    print(f"Best Model: {best_model} (Average Time: {best_model_time:.2f} seconds)")
    print(
        f"Best Resolution: {best_resolution} (Average Time: {best_resolution_time:.2f} seconds)"
    )
else:
    print("No results to display. Benchmark was incomplete.")
