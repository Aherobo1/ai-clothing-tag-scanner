import runpod
import time
import os
import json
from dotenv import load_dotenv
load_dotenv()

runpod.api_key = os.getenv('RUNPOD_API_KEY')

class TagIdentification:
    _instance = None
    _cache = {}
    _cache_file = os.path.join(os.path.dirname(__file__), 'tag_identification_cache.json')
    _is_cache_loaded = False

    def __new__(cls, endpoint_id):
        if cls._instance is None:
            cls._instance = super(TagIdentification, cls).__new__(cls)
            cls._instance.endpoint_id = endpoint_id
            cls._instance.endpoint = runpod.Endpoint(endpoint_id)
            cls._instance._load_cache()
        return cls._instance

    def __init__(self, endpoint_id):
        # Initialization already done in __new__
        pass

    def _load_cache(self):
        """Load the tag identification cache from file."""
        if self._is_cache_loaded:
            return

        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
                print(f"Loaded {len(self._cache)} cached tag identifications")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self._cache = {}

        self._is_cache_loaded = True

    def _save_cache(self):
        """Save the tag identification cache to file."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
            print(f"Saved {len(self._cache)} cached tag identifications")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def preload_tags(self, image_urls):
        """
        Preload tag identifications for a list of image URLs.
        Args:
            image_urls: List of image URLs to preload
        """
        print(f"Preloading {len(image_urls)} tag identifications...")

        for i, url in enumerate(image_urls):
            if url in self._cache:
                continue

            try:
                result = self._identify_tag_api(url)
                if result:
                    self._cache[url] = result

                # Save cache periodically
                if (i + 1) % 10 == 0:
                    self._save_cache()
                    print(f"Preloaded {i+1}/{len(image_urls)} tags")
            except Exception as e:
                print(f"Error preloading tag for {url}: {e}")

        # Final save
        self._save_cache()
        print(f"Completed preloading {len(image_urls)} tags")

    def identify_tag(self, image_url):
        """
        Identify tag from image URL, using cache if available.
        Args:
            image_url: URL of the image to identify
        Returns:
            Tag identification result
        """
        # Check cache first
        if image_url in self._cache:
            print(f"Cache hit for {image_url}")
            return self._cache[image_url]

        # Call API if not in cache
        result = self._identify_tag_api(image_url)

        # Cache the result
        if result:
            self._cache[image_url] = result
            self._save_cache()

        return result

    def _identify_tag_api(self, image_url):
        """Make the actual API call to identify the tag."""
        prompt = """
        You will tell me which tag it belongs to:
        1. Alstyle Apparel & Activewear T-Shirt Tags 1995-2006
        2. Anvil T-Shirt Tags 1989-2007
        3. Ched and Anvil T-Shirt Tags 1976-1988
        4. Delta T-Shirt Tags 1988-2014
        5. Fruit of the Loom 1970-1998
        6. Giant T-Shirt Tags 1991-1996
        7. Gildan T-Shirt Tags 1995-2002
        8. Hanes T-Shirt Tags 1989-1997
        9. Jerzees T-Shirt Tags 1985-1998
        10. Oneita T-Shirt Tags 1984-1999
        11. Screen Stars T-Shirt Tags 1980-1994
        12. Signal T-Shirt Tags 1977-1994
        13. Sportswear T-Shirt Tags 1968 – 1990
        14. Stedman & Hi Cru T-Shirt Tags 1971-1997
        15. Tennessee River T-Shirt Tags 1984-2010
        16. Wild Oats T-Shirt Tags 1984-1997
        17. Winterland T-Shirt Tags 1982-2008
        18. Others

        Just Give me the Tag only(Don't add anything)
        """

        payload = {
            "input": {
                "input_image_url": image_url,
                "vlm_prompt": "What is the tag of the t-shirt? Just give me the name of the tag nothing else",
                "max_new_tokens": 20
            }
        }

        # Send the request to the endpoint and get the response
        run_request = self.endpoint.run(payload)

        # Check the status of the endpoint run request in a loop until completed or an error occurs
        while True:
            status = run_request.status()
            if status == 'COMPLETED':
                return run_request.output()
            elif status == 'FAILED':
                print("Request failed.")
                return None
            time.sleep(1)
        return None 