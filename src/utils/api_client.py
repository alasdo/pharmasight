import requests
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger


class RateLimitedClient:
    def __init__(self, base_url, requests_per_second=2.0, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-Api-Key"] = api_key

    def _wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get(self, endpoint, params=None):
        self._wait()
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download_file(self, url, output_path, chunk_size=8192):
        self._wait()
        logger.info(f"Downloading {url}")
        resp = self.session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    if downloaded % (chunk_size * 100) == 0:
                        logger.info(f"  {pct:.1f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)")
        logger.info(f"  Saved to {output_path}")