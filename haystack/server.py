import http.server
import socketserver
import threading
import webbrowser
import logging
import time

PORT = 8000
DIRECTORY = "."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        logging.info(f"GET request for {self.path}")
        super().do_GET()

    def do_POST(self):
        logging.info(f"POST request for {self.path}")
        super().do_POST()


def run_server(open_browser=True, timeout=300):  # Timeout in seconds
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT} (Press Ctrl+C to terminate the server)")
        if open_browser:
            try:
                webbrowser.open(f"http://localhost:{PORT}/viewer.html")
            except Exception as e:
                logging.error(f"Failed to open browser: {e}")

        httpd.timeout = 1  # Setting a timeout for the server loop
        try:
            while True:
                httpd.handle_request()
        except KeyboardInterrupt:
            print("Server interrupted by user.")
        finally:
            print("Server shutting down...")
            httpd.server_close()


if __name__ == "__main__":
    # Prompt user for optional browser opening
    open_browser = input("Open browser automatically? (y/n): ").strip().lower() == 'y'

    try:
        # Run the server on the main thread to allow Control-C to interrupt
        run_server(open_browser=open_browser)
    except KeyboardInterrupt:
        print("Shutting down the server with Control-C...")
    finally:
        print("Cleanup complete.")