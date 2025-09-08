from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import requests
import time
import json
from datetime import datetime, timedelta, UTC, timezone
import os
from ligo.gracedb.rest import GraceDb
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from io import BytesIO

def plot_source_on_event(source_id, event_id):
    with open("events.json") as f:
        event = json.load(f).get(event_id)
    time_array = np.array(event.get("time_single"))
    mean_preds = np.array(event.get("mean_preds_inverted"))
    uncertainty = np.array(event.get("uncertainty_reshaped"))
    superevent_id = event_id
    time = event.get("time")

    event_t = Time(time, format='isot', scale='utc')
    event_mjd = event_t.mjd

    photometry_url = f"https://fritz.science/api/sources/{source_id}/photometry"
    token = "bb8a0369-068c-4aca-94d7-ee9f97a6a412"
    headers = {"Authorization": f"token {token}"}

    photometry_r = requests.get(photometry_url, headers=headers)

    if photometry_r.status_code != 200:
        print("Request failed")
        return

    photometry_data = photometry_r.json()
    photometry_list = photometry_data.get("data", [])

    # Define colors for plotting
    colors = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'blue'}

    # Time array for plotting
    time_single = time_array

    # Filter names for ZTF filters
    filter_names = ['ztfg', 'ztfr', 'ztfi']

    # Determine the number of examples from mean_preds
    num_examples = len(mean_preds)

    # Loop through all available examples
    for example_idx in range(num_examples):
        # Select one example light curve to plot
        mean_curve_new = mean_preds[example_idx]
        uncertainty_curve_new = uncertainty[example_idx]

        # Create a plot for the predicted light curve and uncertainty
        plt.figure(figsize=(10, 6))

        for i in range(3):  # 3 filters
            # Plot the mean predicted light curve
            plt.plot(time_single, mean_curve_new[:, i], label=f'Predicted {filter_names[i]}', color=colors[filter_names[i]])
            plt.fill_between(time_single, 
                             mean_curve_new[:, i] - 5 * uncertainty_curve_new[:, i], 
                             mean_curve_new[:, i] + 5 * uncertainty_curve_new[:, i], 
                             color=colors[filter_names[i]], alpha=0.2)

        for obj in photometry_list:
            if obj.get("mag") is not None:
                # Plot the observed magnitude
                plt.errorbar(obj['mjd'] - event_mjd, obj['mag'], yerr=obj['magerr'], fmt='o', label=f'Observed {obj["filter"]}', color=colors[obj["filter"]])
            else:
                # Plot the upper limit
                plt.scatter(obj['mjd'] - event_mjd, obj['mag'], marker='^', s=200, color=colors[obj["filter"]])

        # Plot settings
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude AB')
        plt.gca().invert_yaxis()  # Invert the y-axis for magnitude
        plt.legend()
        plt.xlim(0, 6)
        print("me")
        plt.savefig(superevent_id+'.png')

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")  # Save to buffer
        buffer.seek(0)
        plt.show()  # Show the plot after saving
        plt.close()  # Close the figure to free memory

        #post_comment_to_skyportal(time, buffer, superevent_id)
    return

# Function 1: Download skymap from GraceDB
def get_skymap_path(graceid, download_dir="skymaps"):
    """
    Downloads the skymap FITS file for a GCN/GraceDB event.
    
    Parameters:
        graceid (str): GraceDB event ID, e.g., "S230518h"
        download_dir (str): Directory to save the skymap
    
    Returns:
        str: Full path to the downloaded skymap file
    """
    os.makedirs(download_dir, exist_ok=True)
    client = GraceDb()

    files = client.files(graceid)

    url=f"https://gracedb.ligo.org/api/superevents/{graceid}/files/bayestar.fits.gz"
    local_path = os.path.join(download_dir, f"{graceid}_bayestar.fits.gz")
    if not os.path.exists(local_path):
        print(f"Downloading skymap from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    return local_path

    raise FileNotFoundError(f"No skymap file found for event {graceid}")

# Function 2: Check if RA/Dec is inside the 90% credible region
def is_in_90_percent(ra, dec, skymap_path):
    """
    Determines if a given sky position is within the 90% credible region of the skymap.
    
    Parameters:
        ra (float): Right Ascension in degrees
        dec (float): Declination in degrees
        skymap_path (str): Path to the skymap FITS file
    
    Returns:
        bool: True if within 90% credible region, else False
    """
    with fits.open(skymap_path) as hdul:
        prob = hdul[1].data['PROB']
        nside = hp.npix2nside(len(prob))

        # Find the probability threshold that encloses 90% of the total probability
        sorted_prob = np.sort(prob)[::-1]
        cumsum = np.cumsum(sorted_prob)
        level_90 = sorted_prob[np.searchsorted(cumsum, 0.9)]

        # Convert RA, Dec to HEALPix pixel
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        theta = 0.5 * np.pi - coord.dec.radian
        phi = coord.ra.radian
        ipix = hp.ang2pix(nside, theta, phi)

        return prob[ipix] >= level_90



base_dir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{base_dir}/source_ids.txt"

# Only create the file if it doesn't exist
if not os.path.exists(filepath):
    with open(filepath, "w") as f:
        pass  # Creates an empty file


base_url = "https://fritz.science/api"
url = base_url + "/sources"
token = "bb8a0369-068c-4aca-94d7-ee9f97a6a412"
headers = {"Authorization": f"token {token}"}
group_ids = [1544]  # If applicable
max_retries = 3

def id_in_file(string_id, filepath):
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == string_id:
                return True
    return False


def api(method, endpoint, params=None, data=None):
    url = f"{base_url}/{endpoint}"
    return requests.request(method, url, headers=headers, params=params, json=data)

def check_source_with_events(source):
    # Load events
    with open("events.json") as f:
        events = json.load(f)

    now = datetime.now(UTC)

    for event_id, event_info in events.items():
        # Parse event time
        event_time = datetime.fromisoformat(event_info["time"]).replace(tzinfo=timezone.utc)
        age = now - event_time
        if age > timedelta(days=30):
            continue  # Skip old events

        source_time = datetime.fromisoformat(source.get("created_at")).replace(tzinfo=timezone.utc)
        source_id = source.get("id")

        # Check time window
        if not (event_time - timedelta(days=3) <= source_time <= event_time + timedelta(days=7)):
            continue

        # Spatial crossmatch check
        skymap_path = get_skymap_path(event_id)
        print(skymap_path)
        if skymap_path:
            spatial_ok = is_in_90_percent(source.get("ra"), source.get("dec"), skymap_path)

        if spatial_ok:
            print(event_id)
            plot_source_on_event(source_id, event_id)
            return

    # If no match found:
    print("No match")
    return


def task():
    print("Running task at", datetime.now())

scheduler = BlockingScheduler()

# Use a DST-aware timezone
pacific = pytz.timezone("US/Pacific")

trigger = CronTrigger(hour=11, minute=0, timezone=pacific)  # 11:00 AM PT every day

#scheduler.add_job(task, trigger)

#scheduler.start()

# API Code

base_url = "https://fritz.science/api"
url = base_url + "/sources"
token = "bb8a0369-068c-4aca-94d7-ee9f97a6a412"
headers = {"Authorization": f"token {token}"}
group_ids = [1544]  # If applicable
max_retries = 3

params = {
    'pageNumber': 1,
    'numPerPage': 100,
    'group_ids': group_ids,
    'totalMatches': None,
    'useCache': True, # Enable caching
    'queryID': None # Server will return this in the first response
}

all_sources = {}

first_source_found = False

now = datetime.now(UTC)

yes = 0

retries_remaining = max_retries
while retries_remaining > 0:
    r = requests.get(
        url,
        params=params,
        headers=headers,
    )

    if r.status_code == 429:
        print("Request rate limit exceeded; waiting 1s before trying again...")
        time.sleep(1)
        continue

    data = r.json()

    source_list = data["data"].get("sources", [])

    if not source_list:
        break  # No more sources

    if data["status"] == "success":
        retries_remaining = max_retries
    else:
        print(f"Error: {data["message"]}; waiting 5s before trying again...")  # log as appropriate
        retry_attempts -= 1
        time.sleep(5)
        continue

    for src in source_list:
        src_id = src.get("id")
        src_time = src.get("created_at")
        print(now)
        print(src_time)
        if ((now - datetime.fromisoformat(src_time).replace(tzinfo=timezone.utc)) > timedelta(days=30)) or id_in_file(src_id, filepath):
            first_source_found = True
            break
        if yes == 0:
            print(src)
            yes = 1
        if src_id:
            all_sources[src_id] = src

    total_matches = data["data"]["totalMatches"]
    params["queryID"] = data["data"]["queryID"] # Pass the queryID to the next request

    print(f"Fetched {len(all_sources)} of {total_matches} sources.")

    if first_source_found or len(all_sources) >= total_matches:
        break

    params['pageNumber'] += 1

for src_id in all_sources:
    print(f"Checking source {src_id}")
    with open(filepath, "a") as f:
        f.write(f"{src_id}\n")
    check_source_with_events(all_sources[src_id])