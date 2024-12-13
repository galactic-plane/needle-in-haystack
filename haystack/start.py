import os
import json
from gradio_client import Client, handle_file
from PIL import Image, ImageDraw
from datetime import datetime
from alive_progress import alive_bar
import subprocess

# Initialize Gradio client
client = Client("http://127.0.0.1:7860/")

# Directory setup
image_dir = "./images/"
annotated_dir = "./annotated/"
output_json = "image_data.json"
output_html = "viewer.html"
florence_model = "microsoft/Florence-2-base-ft"

# Ensure directories exist
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"The directory {image_dir} does not exist.")

if not os.path.exists(annotated_dir):
    os.makedirs(annotated_dir)

# Clear the annotated directory
for file in os.listdir(annotated_dir):
    file_path = os.path.join(annotated_dir, file)
    if os.path.isfile(file_path):
        os.unlink(file_path)

# Supported image formats
supported_formats = (".jpg", ".jpeg", ".png", ".webp")

# Scan for images in the directory
images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(supported_formats)
]

if not images:
    raise FileNotFoundError(f"No images found in the directory {image_dir}.")

# Define task prompts
task_prompts = ["Caption", "Detailed Caption", "Object Detection"]


def clean_keys(d):
    if isinstance(d, dict):
        return {
            (
                k.replace("<CAPTION>", "Caption")
                .replace("<DETAILED_CAPTION>", "Detailed Caption")
                .replace("<OD>", "Object Detection")
            ): clean_keys(v)
            for k, v in d.items()
        }
    elif isinstance(d, list):
        return [clean_keys(i) for i in d]
    else:
        return d


# Process each image
image_results = []
label_counts = {}
image_labels_map = {}

# Process images with a single progress bar
total_steps = len(images) * len(task_prompts)
with alive_bar(total_steps, title="Processing Images") as bar:
    for image_path in images:
        print(f"Processing: {image_path}")

        # Initialize result dictionary for this image
        result_dict = {}

        # Loop through selected task prompts
        for task_prompt in task_prompts:
            print(f"  Task Prompt: {task_prompt}")
            result = client.predict(
                image=handle_file(image_path),
                task_prompt=task_prompt,
                text_input=None,
                model_id=florence_model,
                api_name="/process_image",
            )
            result_dict[task_prompt] = result[0]
            bar.text = f"Processing {os.path.basename(image_path)}: {task_prompt}"
            bar()

        # Open the image and draw bounding boxes for Object Detection (if applicable)
        if "Object Detection" in result_dict:
            detection_data = json.loads(
                result_dict["Object Detection"].replace("'", '"')
            )
            bboxes = detection_data["<OD>"]["bboxes"]
            labels = detection_data["<OD>"]["labels"]

            # Update label counts for the treemap and image-to-label mapping
            image_labels_map[image_path] = labels
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), label, fill="red")

            # Save annotated image with a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            annotated_filename = f"annotated_{os.path.basename(image_path).split('.')[0]}_{timestamp}.png"
            annotated_path = os.path.join(annotated_dir, annotated_filename)
            img.save(annotated_path)
        else:
            annotated_path = (
                image_path  # Use original image if no detection is performed
            )

        # Clean up and format JSON output
        cleaned_result_dict = {}
        for key, value in result_dict.items():
            try:
                cleaned_result_dict[key] = json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                cleaned_result_dict[key] = value

        combined_json = clean_keys(cleaned_result_dict)  # Clean keys for readability

        # Use "Caption" as the title for the modal
        modal_title = cleaned_result_dict.get("Caption", {}).get("", "Image")

        image_results.append(
            {
                "original_path": image_path,
                "annotated_path": annotated_path,
                "combined_json": combined_json,
                "modal_title": modal_title,
            }
        )

        def reformat_result(result_dict):
            for key in list(result_dict.keys()):
                value = result_dict[key]
                # Check if value is a nested dictionary with an empty string as key
                if isinstance(value, dict) and "" in value:
                    result_dict[key] = value[""]

            # Convert to JSON with proper formatting
            return json.dumps(result_dict, indent=4)


# Prepare the treemap data
treemap_data = {
    "name": "Root",
    "children": [
        {"name": label, "value": count} for label, count in label_counts.items()
    ],
}

# Save the JSON data
data = {"images": image_results}
with open(output_json, "w") as json_file:
    json.dump(data, json_file, indent=2)

print(f"JSON data saved as: {output_json}")

# Generate HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Needle in a Haystack</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" />
    <link rel="icon" href="haystack.png" type="image/png" />
    <style>
      .gallery-img {{
        max-height: 225px;
        object-fit: cover;
        cursor: pointer;
      }}
      pre {{
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        padding: 10px;
        font-size: 14px;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 200px;
        height: 200px;
        text-align: left;
      }}
      .modal-fullscreen {{
        max-width: 98vw;
        max-height: 98vh;
      }}
      .modal-body img {{
        width: 100%;
        height: auto;
      }}
      #treemap-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50vh; /* Full height of the viewport */
        width: 100vw; /* Full width of the viewport */
      }}
      footer {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Adjust as needed */
        text-align: center; /* Ensures the text inside is centered if it's multiline */
        width: 100%; /* Ensures it spans the full width of the viewport */
        background-color: #f8f9fa; /* Optional, adds a background color */
      }}
      #reset-button {{
        margin-top: 20px;
      }}
      #search-box {{
        margin-top: 5px;
        width: 400px;
      }}
    </style>
  </head>
  <body>
    <header>
      <div class="navbar navbar-dark bg-dark shadow-sm">
        <div class="container">
          <a href="#" class="navbar-brand d-flex align-items-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              fill="none"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              aria-hidden="true"
              class="me-2"
              viewBox="0 0 24 24"
            >
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 1 1 2-2h4l2-3h6l2 3h4a2 2 0 1 1 2 2z"></path>
              <circle cx="12" cy="13" r="4"></circle>
            </svg>
            <strong>Needle in a Haystack</strong>
          </a>
          <div class="d-flex justify-content-end py-2">
            <input type="text" id="search-box" class="form-control me-2" placeholder="Search..." />
          </div>
          <button
            class="navbar-toggler collapsed"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarHeader"
            aria-controls="navbarHeader"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>
        </div>
      </div>
      <div class="bg-dark collapse" id="navbarHeader">
        <div class="container">
          <div class="row">
            <div class="col-sm-8 col-md-7 py-4">
              <h4 class="text-white">About</h4>
              <p class="text-white">
                This script leverages the Microsoft Florence-2 API to process images and generate an interactive HTML gallery. It performs tasks such
                as caption generation and object detection, annotating images with bounding boxes and labels for detected objects. The results are
                formatted into a JSON structure and presented alongside annotated images. Using Bootstrap for styling and D3.js for visualization, the
                gallery includes a treemap to represent label occurrences, enabling users to filter images by detected objects. This solution is
                practical for applications in research, presentations, and automated inspection systems, where visualizing AI-driven insights is
                essential.
              </p>
            </div>
            <div class="col-sm-4 offset-md-1 py-4">
              <h4 class="text-white">About Daniel Penrod</h4>
              <ul class="list-unstyled">
                <li><a href="https://github.com/galactic-plane" class="text-white">GitHub</a></li>
                <li><a href="https://www.linkedin.com/in/daniel-penrod-sr" class="text-white">LinkedIn</a></li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </header>
    <main>
      <div class="container">
        <section class="py-5 text-center container">
          <div class="row py-lg-5">
            <div class="col-lg-6 col-md-8 mx-auto">
              <h1 class="fw-light">Needle in a Haystack</h1>
              <br />
              <img src="haystack.png" width="200" height="200" class="img-fluid" alt="Needle in a Haystack" />
              <br /><br />
              <p class="lead text-muted">
                Explore object detection results on your images. Click on any image for detailed visualization. Click items in the treemap below to
                filter the gallery. What will you find?
              </p>
            </div>
          </div>
        </section>
        <section class="album py-5 text-center container">
          <h2 class="fw-light">Object Detection Gallery</h2>
          <button id="reset-button" class="btn btn-primary">Reset Gallery</button>
          <div class="row row-cols-1 row-cols-sm-1 row-cols-md-1 g-3">
            <div id="gallery" class="row row-cols-1 row-cols-sm-2 row-cols-md-3 g-3"></div>
          </div>
          <div class="text-center mt-4">
            <button id="prev-page" class="btn btn-primary page-btn">Previous</button>
            <button id="next-page" class="btn btn-primary page-btn">Next</button>
          </div>
        </section>
        <section class="album py-5 text-center">
          <h2 class="fw-light">Tree Map</h2>
        </section>
      </div>
    </main>
    <div id="treemap-container">
      <svg id="treemap-svg"></svg>
    </div>

    <footer class="text-muted py-5">
      <div class="container">
        <p class="mb-1">Needle in a Haystack &copy; 2024. Created using Python. Author: Daniel Penrod.</p>
      </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <script>
      const jsonFile = "image_data.json";
      const itemsPerPage = 6;
      let currentPage = 1;
      let totalPages = 1;
      let imageData = [];
      var treemapData = {json.dumps(treemap_data)};
      var image_labels_map = {json.dumps(image_labels_map)};
      var width = 800;
      var height = 600;

      var root = d3.hierarchy(treemapData).sum(function (d) {{
        return d.value;
      }});

      d3.treemap().size([width, height]).padding(2)(root);

      var colorScale = d3.scaleOrdinal(d3.schemeCategory10);

      var svg = d3.select("#treemap-svg").attr("width", width).attr("height", height);

      var nodes = svg
        .selectAll("g")
        .data(root.leaves())
        .enter()
        .append("g")
        .attr("transform", function (d) {{
          return "translate(" + d.x0 + "," + d.y0 + ")";
        }});

      nodes
        .append("rect")
        .attr("width", function (d) {{
          return d.x1 - d.x0;
        }})
        .attr("height", function (d) {{
          return d.y1 - d.y0;
        }})
        .attr("fill", function (d) {{
          return colorScale(d.data.name);
        }})
        .attr("stroke", "#fff")
        .style("cursor", "pointer")
        .on("click", function (event, d) {{
          var label = d.data.name;
          filterGalleryByLabel(label);
        }});

      nodes
        .append("text")
        .attr("dx", 5)
        .attr("dy", 15)
        .text(function (d) {{
          return d.data.name + " (" + d.data.value + ")";
        }})
        .attr("fill", "#fff")
        .attr("font-size", "12px");

      // Reset functionality using the reset button
      document.getElementById("reset-button").addEventListener("click", function () {{
        renderGallery();
      }});

      // Search functionality
      document.getElementById("search-box").addEventListener("input", function () {{
        var searchValue = this.value.toLowerCase();
        if (searchValue) {{
          filterGalleryByLabel(searchValue);
        }} else {{
          renderGallery();
        }}
      }});

      function filterGalleryByLabel(label) {{
        const gallery = document.getElementById("gallery");
        gallery.innerHTML = "";

        const filteredImages = imageData.filter((image) => {{
          const labels = (image_labels_map[image.original_path] || []).join(",").toLowerCase();
          return labels.includes(label.toLowerCase());
        }});

        filteredImages.forEach((image, i) => {{
          const col = document.createElement("div");
          col.className = "col";
          col.innerHTML = `
          <div class="col gallery-item" data-labels="${{(image_labels_map[image.original_path] || []).join(",")}}">
            <div class="card shadow-sm">
              <img src="${{
                image.annotated_path
              }}" class="bd-placeholder-img card-img-top gallery-img" alt="Image ${{i}}" data-bs-toggle="modal" data-bs-target="#modal${{i}}">
              <div class="card-body">
                <pre><code class="language-json">${{JSON.stringify(image.combined_json, null, 2)}}</code></pre>
              </div>
            </div>
          </div>
          <div class="modal fade" id="modal${{i}}" tabindex="-1" aria-labelledby="modalLabel${{i}}" aria-hidden="true">
            <div class="modal-dialog modal-fullscreen">
              <div class="modal-content">
                <div class="modal-header">
                  <h5 class="modal-title" id="modalLabel${{i}}">${{(image_labels_map[image.original_path] || []).join(",")}}</h5>
                  <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                  <img src="${{image.annotated_path}}" class="img-fluid" alt="Annotated Image ${{i}}">
                </div>
              </div>
            </div>
          </div>
          `;
          gallery.appendChild(col);
        }});

        hljs.highlightAll();
      }}

      async function fetchData() {{
        const response = await fetch(jsonFile);
        const data = await response.json();
        imageData = data.images;
        totalPages = Math.ceil(imageData.length / itemsPerPage);
        renderGallery();
      }}

      function renderGallery() {{
        const gallery = document.getElementById("gallery");
        gallery.innerHTML = "";
        const startIdx = (currentPage - 1) * itemsPerPage;
        const endIdx = Math.min(startIdx + itemsPerPage, imageData.length);

        for (let i = startIdx; i < endIdx; i++) {{
          const image = imageData[i];
          const col = document.createElement("div");
          col.className = "col";
          col.innerHTML = `
          <div class="col gallery-item" data-labels="${{(image_labels_map[image.original_path] || []).join(",")}}">
            <div class="card shadow-sm">
             <img src="${{
               image.annotated_path
             }}" class="bd-placeholder-img card-img-top gallery-img" alt="Image ${{i}}" data-bs-toggle="modal" data-bs-target="#modal${{i}}">
             <div class="card-body">
                <pre><code class="language-json">${{JSON.stringify(image.combined_json, null, 2)}}</code></pre>
             </div>
            </div>
          </div>
           <div class="modal fade" id="modal${{i}}" tabindex="-1" aria-labelledby="modalLabel${{i}}">
                <div class="modal-dialog modal-fullscreen">
                  <div class="modal-content">
                    <div class="modal-header">
                      <h5 class="modal-title" id="modalLabel${{i}}">${{(image_labels_map[image.original_path] || []).join(",")}}</h5>
                      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                      <img src="${{image.annotated_path}}" class="img-fluid" alt="Annotated Image ${{i}}">
                    </div>
                  </div>
                </div>
              </div>
          `;
          gallery.appendChild(col);
        }}

        hljs.highlightAll();
      }}

      document.getElementById("prev-page").addEventListener("click", () => {{
        if (currentPage > 1) {{
          currentPage--;
          renderGallery();
        }}
      }});

      document.getElementById("next-page").addEventListener("click", () => {{
        if (currentPage < totalPages) {{
          currentPage++;
          renderGallery();
        }}
      }});

      fetchData();

      document.addEventListener("DOMContentLoaded", () => {{
        hljs.highlightAll();
      }});
    </script>
  </body>
</html>
"""

# Save HTML to file
with open(output_html, "w") as html_file:
    html_file.write(html_content)

print(f"HTML gallery saved as: {output_html}")

# Start the server
subprocess.Popen(['start', 'cmd', '/k', 'python server.py'], shell=True)