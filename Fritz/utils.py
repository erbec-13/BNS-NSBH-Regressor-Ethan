import numpy as np
from astropy.table import Table
import requests
from io import BytesIO
import base64
import astropy_healpix as ah
import healpy as hp
import xmltodict
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels
import matplotlib.pyplot as plt
from astropy.time import Time
import os

SKYPORTAL_HOST = os.getenv("SKYPORTAL_HOST", "https://fritz.science")
SKYPORTAL_TOKEN = os.getenv("SKYPORTAL_TOKEN")
if SKYPORTAL_TOKEN is None:
    raise ValueError("Please set the SKYPORTAL_TOKEN environment variable")
HEADERS = {"Authorization": f"token {SKYPORTAL_TOKEN}"}


def parse_gcn(response):
    return xmltodict.parse(response)


def get_params(event_dict):
    params = {item['@name']: item['@value'] for item in event_dict['voe:VOEvent']['What']['Param']}
    
    superevent_id = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'GraceID'][0]
    event_page = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'EventPage'][0]
    alert_type = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'AlertType'][0]
    group = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'Group'][0]
    
    significant = [item['@value'] for item in event_dict['voe:VOEvent']['What']['Param'] if item.get('@name') == 'Significant'][0]
    
    classification = [item for item in event_dict['voe:VOEvent']['What']['Group'] if item.get('@name') == 'Classification']
    properties = [item for item in event_dict['voe:VOEvent']['What']['Group'] if item.get('@name') == 'Properties']
    try:
        prob_bbh = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'BBH'][0])  
    except:
        prob_bbh = 0
    try:
        prob_ter = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'Terrestrial'][0])
    except:
        prob_ter = 1
    try:
        prob_bns = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'BNS'][0])  
    except:
        prob_bns = 0
    try:
        prob_nsbh = float([item['@value'] for item in classification[0]['Param'] if item.get('@name') == 'NSBH'][0])
    except:
        prob_nsbh = 1
    try:
        has_ns = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasNS'][0])
    except:
        has_ns = 0
    try:
        has_remnant = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasRemnant'][0])
    except:
        has_remnant = 0
    try:
        has_mass_gap = float([item['@value'] for item in properties[0]['Param'] if item.get('@name') == 'HasMassGap'][0])
    except:
        has_mass_gap = 0
    try:
        PAstro = 1 - prob_ter
    except:
        PAstro = 1
    far = float(params.get("FAR", 0))
    skymap_url = next(
        item["Param"]["@value"]
        for item in event_dict["voe:VOEvent"]["What"]["Group"]
        if item.get("@name") == "GW_SKYMAP"
    )
    skymap_response = requests.get(skymap_url)
    skymap_bytes = skymap_response.content
    skymap = Table.read(BytesIO(skymap_bytes))

    level, ipix = ah.uniq_to_level_ipix(
        skymap[np.argmax(skymap["PROBDENSITY"])]["UNIQ"]
    )
    ra, dec = ah.healpix_to_lonlat(ipix, ah.level_to_nside(level), order="nested")

    m, _ = read_sky_map(BytesIO(skymap_bytes))
    nside = ah.level_to_nside(level)
    credible_levels = find_greedy_credible_levels(m)
    pixel_area_deg2 = np.sum(credible_levels <= 0.9) * hp.nside2pixarea(
        nside, degrees=True
    )

    longitude = ra.value
    latitude = dec.value
    distmean = skymap.meta.get("DISTMEAN", "error")
    area_90 = pixel_area_deg2

    has_ns = float(properties.get("HasNS", 0))
    has_remnant = float(properties.get("HasRemnant", 0))
    has_mass_gap = float(properties.get("HasMassGap", 0))
    PAstro = 1 - prob_ter

    far_format = 1.0 / (far * 3.15576e7)
    t0 = event_dict["voe:VOEvent"]["WhereWhen"]["ObsDataLocation"][
        "ObservationLocation"
    ]["AstroCoords"]["Time"]["TimeInstant"]["ISOTime"]
    dateobs = Time(t0, precision=0).datetime
    time = dateobs.strftime("%Y-%m-%dT%H:%M:%S")

    skymap_url = [
        item["Param"]["@value"]
        for item in event_dict["voe:VOEvent"]["What"]["Group"]
        if item.get("@name") == "GW_SKYMAP"
    ][0]
    skymap_type = skymap_url.split("files/")[1]

    if alert_type == "Preliminary":
        alert_type = f"{alert_type}-{str(skymap_type[-1])}"

    return (
        superevent_id,
        event_page,
        alert_type,
        group,
        prob_bbh,
        prob_bns,
        prob_nsbh,
        far_format,
        distmean,
        area_90,
        longitude,
        latitude,
        has_ns,
        has_remnant,
        has_mass_gap,
        significant,
        prob_ter,
        skymap,
        PAstro,
        time,
        skymap_type,
    )


def predict_with_uncertainty(model, X, n_iter=1000):
    preds = np.array([model.predict(X, verbose=0) for _ in range(n_iter)])
    mean_preds = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    return mean_preds, uncertainty


def fetch_event_id(time):
    response = requests.get(f"{SKYPORTAL_HOST}/api/gcn_event/{time}", headers=HEADERS)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch event id: {response.status_code} ({response.text})"
        )
    return response.json()["data"]["id"]


def post_comment_to_skyportal(time, buffer: BytesIO, superevent_id, alert_type):
    event_id = fetch_event_id(time)
    body = base64.b64encode(buffer.getvalue()).decode("utf-8")
    files = {
        "text": f"Oracle: {superevent_id} ({alert_type})",
        "attachment": {
            "body": body,
            "name": f"Oracle_{superevent_id}_{alert_type}_lc.png",
        },
    }
    url = f"{SKYPORTAL_HOST}/api/gcn_event/{event_id}/comments"
    response = requests.post(url, json=files, headers=HEADERS)
    if response.status_code == 200:
        print("Comment posted successfully")
    else:
        raise ValueError(
            f"Failed to post comment: {response.status_code} ({response.text})"
        )


def plot_all_light_curves_with_uncertainty(
    time_array, mean_preds, uncertainty, superevent_id, time, alert_type
):
    # Define colors for plotting
    colors = {"ztfg": "green", "ztfr": "red", "ztfi": "blue"}

    # Time array for plotting
    time_single = time_array

    # Filter names for ZTF filters
    filter_names = ["ztfg", "ztfr", "ztfi"]

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
            plt.plot(
                time_single,
                mean_curve_new[:, i],
                label=f"Predicted {filter_names[i]}",
                color=colors[filter_names[i]],
            )
            plt.fill_between(
                time_single,
                mean_curve_new[:, i] - 5 * uncertainty_curve_new[:, i],
                mean_curve_new[:, i] + 5 * uncertainty_curve_new[:, i],
                color=colors[filter_names[i]],
                alpha=0.2,
            )

        # Plot settings
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude AB")
        plt.gca().invert_yaxis()  # Invert the y-axis for magnitude
        plt.legend()
        # plt.show()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")  # Save to buffer
        buffer.seek(0)
        # plt.show()  # Show the plot after saving
        plt.close()  # Close the figure to free memory

        post_comment_to_skyportal(time, buffer, superevent_id, alert_type)
